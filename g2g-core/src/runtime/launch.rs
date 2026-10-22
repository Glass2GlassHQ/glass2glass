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
//! Value grammar (M840): a value may carry spaces by quoting (`"..."` or
//! `'...'`, either anywhere in the fragment: `location=/tmp/"my dir"/a.ts`) or by
//! escaping (`\ `). A `\` escapes a quote, another `\`, a `!`, a `#`, or
//! whitespace; before anything else it stays literal, so a Windows path
//! (`C:\videos\a.ts`) survives unescaped. An enum property's value is checked
//! against its [`PropertySpec::enum_values`](crate::PropertySpec) before the
//! element sees it, and a [`Flags`](crate::PropKind::Flags) property takes a
//! `+`-joined set (`protocols=udp+tcp`), so a bad name names the valid ones.
//! A pad-name suffix on a reference (`t.src_0`) is accepted but ignored (pads are
//! positional).
//!
//! Encoding profiles (M1089): `encodebin profile="<container>:<stream>[:<stream>]"`
//! is a macro too, expanding into the encoder each stream names plus the muxer
//! for the container (`encodebin2` and `transcodebin` are gst's other names for
//! it, the latter being `decodebin ! encodebin`). The container picks the muxer
//! and a stream picks its encoder; anything the profile does not name
//! negotiates. A stream part may also pin what it does care about (M1097): a
//! `width` / `height`, a `framerate`, a sample `rate` or a `channels` each
//! splice the converter that applies it, and a `bitrate` is set on the encoder.
//! Any other field is refused. An uncompressed stream
//! (`audio/x-raw`) inserts no encoder, and a profile with no container at all is
//! just the encoder. A converter goes in ahead of an encoder that does not take
//! what the branch produces (`videoconvert`, or `audioconvert` +
//! `audioresample`), so a profile works off whatever raw form the source has. With several streams the bin's `name=` moves to the muxer,
//! so a second branch reaches it as `e.` the way any fan-in does, and that
//! branch gets the encoder for its own kind of input.
//!
//! Two `key=value` pairs are launch keywords rather than properties: `name=` is
//! the instance name (and the handle pad references resolve against), and
//! `log-category=` (M847) replaces that instance's `G2G_DEBUG` category, leaving
//! the auto `<type>N` naming alone.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::caps::{Caps, CapsSet};
use crate::caps_transform::CapsTransform;
use crate::element::DynAsyncElement;
use crate::fanout::DynMultiOutputSource;
use crate::format_element::CapsConstraint;
use crate::graph::{Graph, GraphError, NodeId, PadId};
use crate::link::LinkPolicy;
use crate::memory::MemoryDomainKind;
use crate::property::{PropError, PropValue, PropertySpec, ValueError};
use crate::runtime::autoplug::{
    is_raw_audio, is_raw_video, ContainerStreams, FallbackSourceRole, FallbacksrcSourceFactory,
    FanoutRebuild, PadKind, PadRequest, PlaybinSinks, PrimaryStream, Registry, RestartPolicy,
    UriError, UriFanout, UriFanoutHead, UriFanoutPort, UriRebuild, UriSource,
};
use crate::runtime::parse_scope::ParseScope;
use crate::runtime::{
    DynSourceLoop, GraphMutator, GraphNode, GraphNodeRef, MutationError, UnblockHandle,
};
use crate::stream::StreamType;

/// Why [`parse_launch`] could not build a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    MalformedProperty {
        element: String,
        token: String,
    },
    /// The element has no property of that name.
    UnknownProperty {
        element: String,
        key: String,
    },
    /// The value did not parse for the property's kind, or was rejected.
    BadValue {
        element: String,
        key: String,
        value: String,
    },
    /// The value names no declared choice of an enum / flag property. `value` is
    /// the offending nick (one entry of a `+`-joined set), or the whole value when
    /// the set was malformed; `values` is the property's declared list.
    BadEnumValue {
        element: String,
        key: String,
        value: String,
        values: &'static str,
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
    /// An output-pad reference (`cap.foo`) names a pad this fan-out element does
    /// not have, or one another link already took (M1241). An inline link or a
    /// bare `cap.` reports an empty `pad`.
    UnknownOutputPad {
        element: String,
        pad: String,
    },
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
    /// An `encodebin` had no `profile=`, or one that is not
    /// `container-caps:stream-caps[:stream-caps...]` (the message says which).
    EncodingProfile(String),
    /// An `encodebin` profile named a container no registered muxer writes, or a
    /// stream codec no compiled-in encoder produces (the message quotes it).
    NoEncoder(String),
    /// An `encodebin` chain's input is not raw video or audio, or its profile has
    /// no stream of that kind, so nothing in the profile can encode it.
    EncodebinInput(String),
    /// A `uridecodebin` / `playbin` was not at the head of its chain. It provides
    /// the source, so it must start the pipeline.
    UriSourceNotAtHead(String),
    /// A `uridecodebin` / `playbin` had no `uri=` property, or a `fallbacksrc`
    /// had neither a `uri=` nor a registered main source.
    MissingUri(String),
    /// The `uri=` could not be turned into a source (bad URI, or no handler
    /// registered for its scheme). The message quotes the URI and reason.
    Uri(String),
    /// Linking two nodes into the graph failed.
    Graph(GraphError),
    /// A `fallbacksrc` asked for `manual-unblock=true`, but the registry carries
    /// no [`UnblockHandle`](crate::runtime::UnblockHandle): nothing could ever
    /// release the source, so the pipeline would deliver nothing.
    MissingUnblockHandle(String),
    /// A `togglerecord` request-pad reference has no partner: a stream through it
    /// needs both its `sink_N` and its `src_N`.
    UnpairedRequestPad {
        element: String,
        pad: String,
    },
    /// A `togglerecord` reference named no request pad, or named stream 0. The
    /// pads are paired, so the stream index has to be explicit, and stream 0 is
    /// the inline keyword's own chain.
    BadRequestPad {
        element: String,
        pad: String,
    },
    /// A one-element description held something other than one element: a `!`,
    /// a pad reference, a caps token, or a second element.
    NotOneElement(String),
    /// A chain description held something other than elements joined by `!`:
    /// a pad reference, a caps token, or a second chain.
    NotElementChain(String),
    /// A one-element description carried `name=` or `log-category=`, which only
    /// a whole launch line can apply.
    LaunchKeyword {
        element: String,
        key: String,
    },
    /// A `decodebin`'s source read its content and recognized no media type
    /// (`filesrc` with an extension it does not know and no `bytestream-format`).
    UnrecognizedContent(String),
    /// A `decodebin`'s container holds no stream the element after it can take,
    /// and `codecs` names the streams its demuxer cannot forward (empty when the
    /// container has no audio or video stream at all).
    NoStreamDecoder {
        location: String,
        codecs: Vec<String>,
    },
    /// `source` is the decodebin's file location, else its upstream element
    NoStreamOfKind {
        source: String,
        kind: StreamType,
    },
    /// A bare `d.` branch of a fan-out found no free stream of a kind its first
    /// element takes.
    NoFanoutStream {
        element: String,
        consumer: String,
    },
    /// A `decodebin name=d` fan-out follows an element that is neither a file of a
    /// container g2g demuxes nor a source whose location a URI fan-out probes (the
    /// element and its location are named).
    DecodebinFanoutUpstream(String),
    // a decodebin's current-video / current-audio names a stream past the last of that kind
    StreamIndexOutOfRange {
        source: String,
        kind: StreamType,
        index: usize,
        count: usize,
    },
    StreamIndexUnsupported(String),
    // a parsebin found no demuxer chain from its input caps to an elementary stream
    NoParseChain(String),
}

impl ParseError {
    // quotes the caps in gst's caps syntax, which a person reads more easily than the Debug form
    pub fn no_decode_chain(input: &Caps) -> Self {
        ParseError::NoDecodeChain(input.to_gst_string())
    }
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
            ParseError::BadEnumValue {
                element,
                key,
                value,
                values,
            } => {
                write!(f, "{element}: invalid value '{value}' for property '{key}'")?;
                if !values.is_empty() {
                    write!(f, " (valid: {values})")?;
                }
                if value.contains('+') {
                    write!(f, " (a flag set joins nicks with '+', e.g. video+audio)")?;
                }
                Ok(())
            }
            ParseError::UnpairedRequestPad { element, pad } => {
                write!(
                    f,
                    "{element}.{pad}: a stream needs both its sink_N and its src_N"
                )
            }
            ParseError::BadRequestPad { element, pad } => {
                write!(
                    f,
                    "{element}.{pad}: name a request pad as sink_N / src_N with N above 0 (stream 0 is the inline {TOGGLERECORD} itself)"
                )
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
            ParseError::UnknownOutputPad { element, pad } => {
                write!(
                    f,
                    "{element}: no free output pad named '{pad}' (the element has no such pad, or another link already took it)"
                )
            }
            ParseError::QueueRole(n) => {
                write!(f, "{n}: a queue must sit between two elements (1-in/1-out); it maps to an edge policy, not a source/sink/branch")
            }
            ParseError::DecodebinNoUpstream => {
                write!(f, "decodebin has no upstream element to decode; it must follow a source or element with declared caps")
            }
            ParseError::NoStreamDecoder { location, codecs } if codecs.is_empty() => {
                write!(f, "decodebin: {location} has no audio or video stream to decode")
            }
            ParseError::NoStreamDecoder { location, codecs } => {
                write!(f, "decodebin: no decoder for {} in {location}", codecs.join(", "))
            }
            ParseError::NoStreamOfKind { source, kind } => {
                write!(f, "decodebin: {source} has no {} stream to decode", kind.name())
            }
            ParseError::NoFanoutStream { element, consumer } => {
                write!(f, "{element}: no free stream of a kind {consumer} takes")
            }
            ParseError::DecodebinFanoutUpstream(upstream) => write!(
                f,
                "decodebin name=d fans out behind a file of a container g2g demuxes or an HLS playlist source, not behind {upstream} (for an rtsp:// stream use uridecodebin uri=... name=d or rtspsrc name=s)"
            ),
            ParseError::NoDecodeChain(caps) => {
                write!(f, "decodebin: no decoder chain from {caps} to raw (no decoder feature compiled in, or a container that needs a demuxer)")
            }
            ParseError::EncodingProfile(msg) => write!(
                f,
                "encodebin profile: {msg} (expected profile=\"container-caps:stream-caps[:stream-caps]\", e.g. profile=\"video/x-matroska:video/x-vp8:audio/x-opus\")"
            ),
            ParseError::NoEncoder(msg) => write!(f, "encodebin: {msg}"),
            ParseError::EncodebinInput(msg) => write!(f, "encodebin: {msg}"),
            ParseError::UriSourceNotAtHead(n) => {
                write!(f, "{n}: provides the source, so it must start the pipeline (be the first element)")
            }
            ParseError::MissingUri(n) => write!(f, "{n}: missing required 'uri=' property"),
            ParseError::Uri(msg) => write!(f, "uri error: {msg}"),
            ParseError::Graph(e) => write!(f, "graph link error: {e:?}"),
            ParseError::MissingUnblockHandle(n) => write!(
                f,
                "{n}: manual-unblock=true needs an UnblockHandle on the registry (Registry::register_unblock_handle), or nothing can release the source"
            ),
            ParseError::NotOneElement(description) => {
                write!(f, "'{description}' is not a single element")
            }
            ParseError::NotElementChain(description) => {
                write!(f, "'{description}' is not a chain of elements joined by '!'")
            }
            ParseError::LaunchKeyword { element, key } => write!(
                f,
                "{element}: '{key}=' is not accepted here, the running graph names the element"
            ),
            ParseError::UnrecognizedContent(n) => write!(
                f,
                "{n}: unrecognized content, no media type matches its leading bytes (name the demuxer or set bytestream-format)"
            ),
            ParseError::StreamIndexOutOfRange {
                source,
                kind,
                index,
                count,
            } => write!(
                f,
                "current-{}={index}, but {source} has {count} {} stream(s)",
                kind.name(),
                kind.name()
            ),
            ParseError::StreamIndexUnsupported(message) => f.write_str(message),
            ParseError::NoParseChain(caps) => write!(
                f,
                "parsebin: no demuxer chain from {caps} to an elementary stream"
            ),
        }
    }
}

/// One parsed element: factory name plus its `key=value` properties (all owned so
/// errors can name them), and the optional `name=` handle that pad references
/// resolve against. `name` and `log-category` are special-cased here (launch
/// keywords, not properties), never applied as properties.
struct ElementSpec {
    name: String,
    props: Vec<(String, String)>,
    instance: Option<String>,
    log_category: Option<String>,
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
    /// `name` is the instance name the expansion chose for it (M1164), `None`
    /// for a node no one addresses.
    Prebuilt {
        node: PrebuiltNode,
        name: Option<String>,
    },
}

/// A node a macro expansion built ahead of the structural pass: a source
/// constructed from a `uri=` scheme handler, or a decoder the auto-plug search
/// instantiated. Spliced into the graph as-is (it has no name to build by).
enum PrebuiltNode {
    Source(Box<dyn DynSourceLoop>),
    Element(Box<dyn DynAsyncElement>),
    UriFanout(UriFanout),
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
/// `{ Text, 0 }`, `"caption"` -> `{ Text, 0 }`, `"src_2"` -> `{ Any, 2 }`. A bare `d.` (empty suffix) or an
/// unrecognized prefix is `{ Any, ordinal }`, i.e. positional by reference order.
fn parse_pad_request(pad: &str, ordinal: usize) -> PadRequest {
    let (prefix, index) = match pad.rsplit_once('_') {
        Some((p, n)) => (p, n.parse::<usize>().ok()),
        None => (pad, None),
    };
    let kind = match prefix {
        "video" => PadKind::Video,
        "audio" => PadKind::Audio,
        // `caption` is the name a closed-caption fan-in gives its second pad; it
        // rides the text kind, which is the one every sidecar-cue pad uses.
        "text" | "subtitle" | "caption" => PadKind::Text,
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
    knows_element: &dyn Fn(&str) -> bool,
) -> Result<ElementSpec, ParseError> {
    let mut spec = ElementSpec {
        name: name.to_string(),
        props: Vec::new(),
        instance: None,
        log_category: None,
    };
    while let Some(&tok) = tokens.peek() {
        if tok == "!" || is_caps_token(tok) || as_ref_name(tok).is_some() {
            break;
        }
        // A bare token naming a registered element opens a new top-level chain
        // rather than being this element's property, the way gst-launch parses
        // `videotestsrc ! xvimagesink audiotestsrc ! pulsesink`. One that names
        // nothing registered is the typo it looks like.
        if !tok.contains('=') && knows_element(tok) {
            break;
        }
        let (key, value) = tok
            .split_once('=')
            .ok_or_else(|| ParseError::MalformedProperty {
                element: name.to_string(),
                token: tok.to_string(),
            })?;
        tokens.next();
        let value = unquote_value(value);
        if key == "name" {
            spec.instance = Some(value);
        } else if key == "log-category" {
            // M847: a launch keyword like `name=`, not a property: it renames this
            // instance's `G2G_DEBUG` filter key rather than configuring it.
            spec.log_category = Some(value);
        } else {
            spec.props.push((key.to_string(), value));
        }
    }
    Ok(spec)
}

/// The characters a `\` may escape in a launch line: the quote characters, a `\`
/// itself, the `!` separator, the `#` comment marker, and whitespace. A `\`
/// before anything else is literal, so an unescaped Windows path
/// (`location=C:\videos\a.ts`) survives.
fn is_escapable(c: char) -> bool {
    matches!(c, '"' | '\'' | '\\' | '!' | '#') || c.is_whitespace()
}

/// Resolve a token's quoting into the literal property value: drop the quote
/// characters that open / close a quoted region and resolve `\x` escapes. A value
/// may open and close a quoted region more than once
/// (`location=/tmp/"my dir"/a.ts`), so this walks the whole token rather than
/// stripping one surrounding pair. Both quote characters work; gst-launch quotes
/// only with `"` (a `'` is literal there), so this accepts a superset.
fn unquote_value(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    let mut quote: Option<char> = None;
    let mut chars = v.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(n) if is_escapable(n) => out.push(n),
                Some(n) => {
                    out.push('\\');
                    out.push(n);
                }
                None => out.push('\\'),
            },
            '"' | '\'' if quote.is_none() => quote = Some(c),
            _ if Some(c) == quote => quote = None,
            _ => out.push(c),
        }
    }
    out
}

/// Split a pipeline string into tokens, honoring quoted property values and
/// `#` comments. Outside quotes, whitespace separates tokens and `!` is a
/// standalone token; inside a `"..."` or `'...'` region both are literal, so a
/// value may contain spaces (and even `!`), e.g. `element="x264enc bitrate=4000"`
/// or `location='/my file.ts'`. A `\` escapes the next character when it is one
/// of the launch-special ones ([`is_escapable`]), so `location=/my\ file.ts` and
/// a `\"` inside a quoted region are literal. A `#` outside quotes starts a
/// comment that runs to end of line (a pasted multi-line pipeline may carry
/// them). Quotes and escapes are kept on the token; [`unquote_value`] resolves
/// them once the `key=` split is known. An unterminated quote runs to end of
/// input (best-effort; the property parse then reports any resulting malformed
/// token).
fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    // The open quote char (`"` or `'`), or `None` outside a quoted region.
    let mut quote: Option<char> = None;
    let mut in_comment = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if in_comment {
            // A comment runs to end of line; the newline (whitespace) ends it.
            if c == '\n' {
                in_comment = false;
            }
            continue;
        }
        // An escape pair stays whole (and inert) through tokenization: neither
        // char can close a quote, split a token, or start a comment.
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                if is_escapable(next) {
                    chars.next();
                    cur.push('\\');
                    cur.push(next);
                    continue;
                }
            }
            cur.push('\\');
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
#[cfg(test)]
fn parse_chains(pipeline: &str) -> Result<Vec<Chain>, ParseError> {
    // The unit tests exercise the tokenizer alone, with nothing registered: every
    // bare token is then a property, which is the pre-registry behaviour.
    parse_chains_with(pipeline, &|_| false)
}

/// Split a `gst-launch` pipeline string into chains, asking `knows_element`
/// whether a bare token opens a new chain (see [`consume_element`]).
fn parse_chains_with(
    pipeline: &str,
    knows_element: &dyn Fn(&str) -> bool,
) -> Result<Vec<Chain>, ParseError> {
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
                        log_category: None,
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
                    cur.push(Item::Element(consume_element(
                        tok,
                        &mut tokens,
                        knows_element,
                    )?));
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

/// The property surface a launch node exposes, so one applier serves every node
/// shape (source, transform / sink, muxer, demuxer, fan-out source) instead of a
/// copy of the parse-and-set loop per shape.
pub trait PropTarget {
    fn specs(&self) -> &'static [PropertySpec];
    fn set(&mut self, name: &str, value: PropValue) -> Result<(), PropError>;
}

macro_rules! impl_prop_target {
    ($($t:ty),* $(,)?) => {
        $(impl PropTarget for Box<$t> {
            fn specs(&self) -> &'static [PropertySpec] {
                (**self).properties()
            }
            fn set(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
                (**self).set_property(name, value)
            }
        })*
    };
}

impl_prop_target!(
    dyn DynSourceLoop,
    dyn DynAsyncElement,
    dyn crate::runtime::DynMultiInputElement,
    dyn crate::runtime::DynMultiOutputElement,
    dyn crate::fanout::DynMultiOutputSource,
);

/// Apply parsed `key=value` props to a node: look each name up in the element's
/// declared [`PropertySpec`]s, parse the text for that spec (which validates an
/// enum nick / flag set against its declared values), then set it.
pub fn apply_props<T: PropTarget>(
    target: &mut T,
    name: &str,
    props: &[(String, String)],
) -> Result<(), ParseError> {
    for (key, raw) in props {
        let bad_value = || ParseError::BadValue {
            element: name.into(),
            key: key.clone(),
            value: raw.clone(),
        };
        let spec = target.specs().iter().find(|s| s.name == key).copied();
        let value = match spec {
            Some(spec) => spec.parse_value(raw).map_err(|e| match e {
                ValueError::Kind(_) => bad_value(),
                ValueError::Nick(nick) => ParseError::BadEnumValue {
                    element: name.into(),
                    key: key.clone(),
                    value: nick,
                    values: spec.enum_values.unwrap_or(""),
                },
            })?,
            // The element takes names it cannot declare, so there is no kind to
            // parse for: the text goes on as written for whatever does know it.
            None if crate::property::takes_undeclared_properties(target.specs()) => {
                PropValue::Str(raw.clone())
            }
            None => {
                return Err(ParseError::UnknownProperty {
                    element: name.into(),
                    key: key.clone(),
                })
            }
        };
        target.set(key, value).map_err(|_| bad_value())?;
    }
    Ok(())
}

/// Build the runnable [`Graph`] from parsed chains: flatten elements, resolve
/// `t.` references into directed links, derive each element's role (and any tee's
/// fan-out width) from its link degree, then construct and wire the nodes.
/// `decodebin` and `parsebin`: not elements, but macros that expand, at parse
/// time, into the chain the auto-plug search finds from their upstream caps:
/// down to raw video or audio through decoders (M193), or for `parsebin` down to
/// the parsed elementary stream, with no decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoplugBin {
    Decode,
    Parse,
}

fn autoplug_bin(name: &str) -> Option<AutoplugBin> {
    match name {
        "decodebin" => Some(AutoplugBin::Decode),
        "parsebin" => Some(AutoplugBin::Parse),
        _ => None,
    }
}

// a parsebin plugs a demuxer while its input is still a container, then the stream's parser
fn parsebin_chain(
    registry: &Registry,
    input: &Caps,
    avoided: &[&str],
) -> Result<Vec<Box<dyn DynAsyncElement>>, ParseError> {
    let elementary = |caps: &Caps| !matches!(caps, Caps::ByteStream { .. });
    let mut chain = registry
        .autoplug_avoiding(
            input,
            &elementary,
            DECODEBIN_MAX_DEPTH,
            MemoryDomainKind::System,
            avoided,
        )
        .ok_or_else(|| ParseError::NoParseChain(input.to_gst_string()))?;
    if chain.is_empty() {
        if let Some(name) = registry.parser_name(input) {
            chain.push(build_element(registry, name, &[])?);
        }
    }
    Ok(chain)
}

// a parsebin port goes through its stream's parser, if one is registered
fn plug_port_parse(
    registry: &Registry,
    graph: &mut Graph<GraphNode>,
    from: PadId,
    to: PadId,
    port: &UriFanoutPort,
    policy: LinkPolicy,
    capacity: Option<usize>,
) -> Result<(), ParseError> {
    let parser = registry
        .parser_name(&port.caps)
        .map(|name| build_element(registry, name, &[]))
        .transpose()?;
    let into_branch = match parser {
        Some(parser) => {
            let node = graph.add_transform(GraphNodeRef::Element(parser));
            graph.link(from, node)?;
            PadId::from(node)
        }
        None => from,
    };
    graph.link_full(into_branch, to, policy, capacity)?;
    Ok(())
}

/// Depth bound for the decodebin auto-plug search: a parse + decode (+ a spare
/// hop) is 2-3, so this leaves headroom without letting an unsatisfiable target
/// wander.
const DECODEBIN_MAX_DEPTH: usize = 6;

/// The decode chain to raw for `input`, decoding into `preferred` (the memory
/// the element right after the macro takes) and into caps `consumer` accepts.
///
/// The chain is returned built, not named: the caps the search chose a decoder
/// to produce reach the element only through the factory that takes them, and a
/// parameterless launch factory would build the decoder's default format again.
/// That is how `decodebin ! wgpusink` reached a strict-NV12 sink with a decoder
/// left on I420; the solver re-fixates among the decoder's advertised formats
/// once the chosen one arrives.
fn autoplug_for_consumer(
    registry: &Registry,
    source: &str,
    input: &Caps,
    preferred: MemoryDomainKind,
    consumer: Option<&dyn DynAsyncElement>,
    avoided: &[&str],
) -> Result<Vec<Box<dyn DynAsyncElement>>, ParseError> {
    let raw = |c: &Caps| is_raw_video(c) || is_raw_audio(c);
    let consumer_constraint = consumer.map(|element| element.caps_constraint_as_transform());
    let consumer_accepts = |c: &Caps| raw_for_consumer(c, consumer_constraint.as_ref());
    let names = |target: &dyn Fn(&Caps) -> bool| {
        registry.autoplug_names_avoiding(input, target, DECODEBIN_MAX_DEPTH, preferred, avoided)
    };
    let no_decode_chain = || ParseError::no_decode_chain(input);
    let raw_names = names(&raw).ok_or_else(no_decode_chain)?;
    // the consumer picks only the output caps, so a pinned decoder or a demux's default stream stays
    let use_consumer_chain = names(&consumer_accepts) == Some(raw_names);
    let consumer_kind = consumer_constraint.as_ref().and_then(consumer_stream_type);
    if let Some(kind) = consumer_kind.filter(|_| !use_consumer_chain) {
        let raw_of_kind = |c: &Caps| raw(c) && StreamType::of(c) == kind;
        if names(&raw_of_kind).is_none() {
            return Err(ParseError::NoStreamOfKind {
                source: source.to_string(),
                kind,
            });
        }
    }
    let target: &dyn Fn(&Caps) -> bool = if use_consumer_chain {
        &consumer_accepts
    } else {
        &raw
    };
    let mut chain = registry
        .autoplug_avoiding(input, target, DECODEBIN_MAX_DEPTH, preferred, avoided)
        .ok_or_else(no_decode_chain)?;
    // M421/M676: the caps-identity parser is invisible to the shortest-chain
    // search, so it never appears in the chain.
    registry.maybe_prepend_parser(input, &mut chain);
    Ok(chain)
}

// the one media kind a consumer takes, `None` when it takes both or its constraint does not say
fn consumer_stream_type(constraint: &CapsConstraint<'_>) -> Option<StreamType> {
    let inputs: Vec<&Caps> = match constraint {
        CapsConstraint::DerivedFields(CapsTransform::RawVideo { .. }) => {
            return Some(StreamType::Video)
        }
        CapsConstraint::DerivedFields(CapsTransform::Audio { .. }) => {
            return Some(StreamType::Audio)
        }
        CapsConstraint::Accepts(set) | CapsConstraint::Identity(set) => {
            set.alternatives().iter().collect()
        }
        CapsConstraint::Mapping(pairs) => pairs
            .iter()
            .flat_map(|(input, _)| input.alternatives())
            .collect(),
        _ => return None,
    };
    let kind = StreamType::of(inputs.first()?);
    let one_media_kind = matches!(kind, StreamType::Video | StreamType::Audio)
        && inputs.iter().all(|caps| StreamType::of(caps) == kind);
    one_media_kind.then_some(kind)
}

fn raw_for_consumer(caps: &Caps, consumer_constraint: Option<&CapsConstraint<'_>>) -> bool {
    (is_raw_video(caps) || is_raw_audio(caps))
        && consumer_constraint.is_none_or(|constraint| constraint.accepts(caps))
}

// the first stream with a decode chain to caps the consumer accepts, or for a parsebin the
// first stream the consumer takes as it is
fn choose_container_stream(
    registry: &Registry,
    location: &str,
    mut streams: ContainerStreams,
    bin: AutoplugBin,
    preferred: MemoryDomainKind,
    consumer: Option<&dyn DynAsyncElement>,
    avoided: &[&str],
) -> Result<PrimaryStream, ParseError> {
    let consumer_constraint = consumer.map(|element| element.caps_constraint_as_transform());
    let consumer_accepts = |c: &Caps| raw_for_consumer(c, consumer_constraint.as_ref());
    let reaches_consumer = |stream: &PrimaryStream| match bin {
        AutoplugBin::Decode => registry
            .autoplug_names_avoiding(
                &stream.caps,
                &consumer_accepts,
                DECODEBIN_MAX_DEPTH,
                preferred,
                avoided,
            )
            .is_some(),
        AutoplugBin::Parse => consumer_constraint
            .as_ref()
            .is_none_or(|constraint| constraint.accepts(&stream.caps)),
    };
    let chosen = match streams.streams.iter().position(reaches_consumer) {
        Some(index) => index,
        // a dropped stream may be the one the consumer wanted, so name it rather than plug another
        None if streams.streams.is_empty() || !streams.unsupported.is_empty() => {
            return Err(ParseError::NoStreamDecoder {
                location: location.to_string(),
                codecs: streams.unsupported,
            });
        }
        // the decoder search after this reports the missing chain
        None => {
            let consumer_kind = consumer_constraint.as_ref().and_then(consumer_stream_type);
            streams
                .streams
                .iter()
                .position(|stream| Some(StreamType::of(&stream.caps)) == consumer_kind)
                .unwrap_or(0)
        }
    };
    Ok(streams.streams.swap_remove(chosen))
}

// named after gst playbin's, and checked here since the decodebin keyword is no element
const DECODEBIN_PROPS: &[&str] = &["current-video", "current-audio"];

// the stream of each kind a decodebin decodes, `None` where gst's -1 leaves the choice to it
#[derive(Debug, Clone, Copy, Default)]
struct StreamIndexes {
    video: Option<usize>,
    audio: Option<usize>,
}

impl StreamIndexes {
    fn of(self, kind: StreamType) -> Option<usize> {
        match kind {
            StreamType::Video => self.video,
            StreamType::Audio => self.audio,
            StreamType::Text | StreamType::Unknown => None,
        }
    }

    fn any_set(self) -> bool {
        self.video.is_some() || self.audio.is_some()
    }
}

fn decodebin_stream_indexes(spec: &ElementSpec) -> Result<StreamIndexes, ParseError> {
    if let Some((key, _)) = spec
        .props
        .iter()
        .find(|(k, _)| !DECODEBIN_PROPS.contains(&k.as_str()))
    {
        return Err(ParseError::UnknownProperty {
            element: spec.name.clone(),
            key: key.clone(),
        });
    }
    Ok(StreamIndexes {
        video: keyword_stream_index(spec, "current-video")?,
        audio: keyword_stream_index(spec, "current-audio")?,
    })
}

// gst types these as gint, -1 to G_MAXINT
fn keyword_stream_index(spec: &ElementSpec, key: &str) -> Result<Option<usize>, ParseError> {
    let Some(text) = prop(spec, key) else {
        return Ok(None);
    };
    let bad_value = || ParseError::BadValue {
        element: spec.name.clone(),
        key: key.to_string(),
        value: text.to_string(),
    };
    let index: i32 = text.parse().map_err(|_| bad_value())?;
    if index == -1 {
        return Ok(None);
    }
    usize::try_from(index).map(Some).map_err(|_| bad_value())
}

// narrows the listed streams to the one of each kind an index names, counted the way ffprobe counts
fn streams_at_indexes(
    location: &str,
    streams: ContainerStreams,
    indexes: StreamIndexes,
) -> Result<ContainerStreams, ParseError> {
    if !indexes.any_set() {
        return Ok(streams);
    }
    // a stream the demux drops still takes a place in the container's order
    if !streams.unsupported.is_empty() {
        return Err(ParseError::StreamIndexUnsupported(alloc::format!(
            "{location} carries streams g2g does not demux ({}), so a stream index cannot be counted",
            streams.unsupported.join(", ")
        )));
    }
    let mut chosen = Vec::new();
    for kind in [StreamType::Video, StreamType::Audio] {
        let Some(index) = indexes.of(kind) else {
            continue;
        };
        let of_kind: Vec<&PrimaryStream> = streams
            .streams
            .iter()
            .filter(|stream| StreamType::of(&stream.caps) == kind)
            .collect();
        let stream = *of_kind
            .get(index)
            .ok_or_else(|| ParseError::StreamIndexOutOfRange {
                source: location.to_string(),
                kind,
                index,
                count: of_kind.len(),
            })?;
        // the demux emits the first stream its properties match
        let selects_earlier_stream = of_kind[..index]
            .iter()
            .any(|earlier| earlier.demux == stream.demux && earlier.props == stream.props);
        if selects_earlier_stream {
            return Err(ParseError::StreamIndexUnsupported(alloc::format!(
                "{} selects streams by codec, so {location}'s {} stream {index} cannot be told from an earlier one",
                stream.demux,
                kind.name()
            )));
        }
        chosen.push(stream.clone());
    }
    Ok(ContainerStreams {
        streams: chosen,
        unsupported: Vec::new(),
    })
}

// an input no container hook lists is the one stream its caps describe
fn check_single_stream_index(
    source: &str,
    caps: &Caps,
    indexes: StreamIndexes,
) -> Result<(), ParseError> {
    if !indexes.any_set() {
        return Ok(());
    }
    let input_kind = StreamType::of(caps);
    if !matches!(input_kind, StreamType::Video | StreamType::Audio) {
        return Err(ParseError::StreamIndexUnsupported(alloc::format!(
            "g2g cannot list the streams of {source}, so it cannot pick one by index"
        )));
    }
    for kind in [StreamType::Video, StreamType::Audio] {
        let Some(index) = indexes.of(kind) else {
            continue;
        };
        let count = usize::from(kind == input_kind);
        if index >= count {
            return Err(ParseError::StreamIndexOutOfRange {
                source: source.to_string(),
                kind,
                index,
                count,
            });
        }
    }
    Ok(())
}

fn referenced_names(chains: &[Chain]) -> alloc::collections::BTreeSet<String> {
    chains
        .iter()
        .flatten()
        .filter_map(|item| match item {
            Item::Ref { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn is_referenced(spec: &ElementSpec, referenced: &alloc::collections::BTreeSet<String>) -> bool {
    spec.instance
        .as_deref()
        .is_some_and(|name| referenced.contains(name))
}

/// Expand every `decodebin` node into the decoder chain the registry auto-plugs
/// from its predecessor's declared caps down to raw (video or audio). An empty
/// chain (the input is already raw) drops the node entirely, so its predecessor
/// links straight to its consumer. The predecessor is the element immediately
/// before the `decodebin` in the same chain; a `decodebin` with no upstream
/// element (chain head, or after a bare `name.` reference) is a loud error,
/// since it has nothing to take its input caps from.
fn expand_decodebin(
    registry: &Registry,
    chains: Vec<Chain>,
    avoided: &[&str],
) -> Result<Vec<Chain>, ParseError> {
    // Names referenced as `name.` somewhere: a `decodebin name=d` with such refs is
    // a FAN-OUT node (M482), not the inline linear case, so it is left unexpanded
    // here and handled by the decodebin-select path in `build_graph` (which probes
    // the file, demuxes, and decodes each requested port). Only the unreferenced
    // inline `... ! decodebin ! ...` expands to a linear decode chain below.
    let referenced = referenced_names(&chains);
    let is_fanout_decodebin =
        |spec: &ElementSpec| autoplug_bin(&spec.name).is_some() && is_referenced(spec, &referenced);

    let mut out = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut new_chain: Chain = Vec::with_capacity(chain.len());
        // The element (name + props) whose output caps feed the next decodebin:
        // the most recent real element. A `Ref` clears it (its caps live in
        // another chain). Props matter because they can re-type the output (a
        // `filesrc`'s `bytestream-format` selects the container).
        let mut upstream: Option<(String, Vec<(String, String)>)> = None;
        // the element before `upstream`, the file an explicit demuxer reads
        let mut before_upstream: Option<(String, Vec<(String, String)>)> = None;
        let mut items = chain.into_iter().peekable();
        while let Some(item) = items.next() {
            match item {
                // A fan-out `decodebin name=d` is left for `build_graph`'s
                // decodebin-select path; it is not linearly expandable.
                Item::Element(spec) if is_fanout_decodebin(&spec) => {
                    if decodebin_stream_indexes(&spec)?.any_set() {
                        return Err(ParseError::StreamIndexUnsupported(alloc::format!(
                            "a {name} fan-out exposes every stream on its own branch, so current-video and current-audio apply to an inline {name} only",
                            name = spec.name
                        )));
                    }
                    before_upstream = None;
                    upstream = Some((spec.name.clone(), spec.props.clone()));
                    new_chain.push(Item::Element(spec));
                }
                Item::Element(spec) if autoplug_bin(&spec.name).is_some() => {
                    let bin = autoplug_bin(&spec.name).expect("the arm matched an auto-plug bin");
                    let indexes = decodebin_stream_indexes(&spec)?;
                    let (pred, props) = upstream.as_ref().ok_or(ParseError::DecodebinNoUpstream)?;
                    let caps = resolve_upstream_caps(registry, pred, props)?;
                    // M1018: the element right after the `decodebin` decides what
                    // memory the decoder should decode into, so a GPU-resident
                    // consumer gets a decoder that decodes into its domain rather
                    // than one whose frames have to be downloaded. Only the
                    // immediate consumer counts, the rule the graph-side
                    // derivation follows.
                    let preferred = match items.peek() {
                        Some(Item::Element(consumer)) => {
                            registry.declared_memory_preference(&consumer.name)
                        }
                        _ => MemoryDomainKind::System,
                    };
                    let consumer = match items.peek() {
                        Some(Item::Element(spec)) => registry
                            .make_element(&spec.name)
                            .map(|mut element| {
                                apply_props(&mut element, &spec.name, &spec.props).map(|()| element)
                            })
                            .transpose()?,
                        _ => None,
                    };
                    // M746: a single-stream demux fixes its output pad before parsing
                    // any byte, so it defaults to a video port; on an audio-only
                    // container the default auto-plug would pick a video decoder and
                    // fail "no caps overlap". If a primary-stream hook sniffs the file
                    // and lists its streams, plug that demux with the selection of the
                    // stream the consumer can take and auto-plug the decoder from its
                    // elementary caps instead.
                    let location = props
                        .iter()
                        .find(|(k, _)| k == "location")
                        .map(|(_, v)| v.as_str());
                    let source = location.unwrap_or(pred.as_str());
                    // `filesrc location=X ! <demux> ! decodebin` reads X through the
                    // demux the line names, so it forwards the stream
                    // `filesrc location=X ! decodebin` would
                    let named_demux_file = match (location, &before_upstream) {
                        (None, Some((file, file_props))) if props.is_empty() => {
                            demuxed_file_streams(registry, file, file_props)?
                        }
                        _ => None,
                    };
                    let named_demux_stream = match named_demux_file {
                        Some((file_location, streams)) => Some(choose_container_stream(
                            registry,
                            &file_location,
                            streams_at_indexes(&file_location, streams, indexes)?,
                            bin,
                            preferred,
                            consumer.as_deref(),
                            avoided,
                        )?),
                        None => None,
                    }
                    .filter(|stream| stream.demux == pred);
                    let demux_in_line = named_demux_stream.is_some();
                    let listed = location.and_then(|location| {
                        registry
                            .container_streams(location, &caps)
                            .map(|streams| (location, streams))
                    });
                    let primary = match (named_demux_stream, listed) {
                        (Some(stream), _) => Some(stream),
                        (None, Some((location, streams))) => Some(choose_container_stream(
                            registry,
                            location,
                            streams_at_indexes(location, streams, indexes)?,
                            bin,
                            preferred,
                            consumer.as_deref(),
                            avoided,
                        )?),
                        (None, None) => {
                            check_single_stream_index(source, &caps, indexes)?;
                            None
                        }
                    };
                    // a parsebin hands on coded pictures, which the crop cannot cut
                    let crops_to_clean_aperture = bin == AutoplugBin::Decode
                        && primary
                            .as_ref()
                            .is_some_and(|stream| stream.has_clean_aperture);
                    let chain_input = match primary {
                        Some(primary) if demux_in_line => {
                            if let Some(Item::Element(demux)) = new_chain.last_mut() {
                                demux.props = primary.props.clone();
                            }
                            primary.caps
                        }
                        Some(primary) => {
                            new_chain.push(Item::Element(ElementSpec {
                                name: primary.demux.to_string(),
                                props: primary.props.clone(),
                                instance: None,
                                log_category: None,
                            }));
                            primary.caps
                        }
                        None => caps,
                    };
                    let decoders = match bin {
                        AutoplugBin::Decode => autoplug_for_consumer(
                            registry,
                            source,
                            &chain_input,
                            preferred,
                            consumer.as_deref(),
                            avoided,
                        )?,
                        AutoplugBin::Parse => parsebin_chain(registry, &chain_input, avoided)?,
                    };
                    // Built here rather than named, so each element is
                    // constructed for the caps the search chose it to produce (a
                    // multi-format decoder emits the format the consumer takes,
                    // not the first one it lists). A pre-built node carries no
                    // name, so `upstream` clears as it does after a
                    // `uridecodebin`.
                    for element in decoders {
                        new_chain.push(Item::Prebuilt {
                            node: PrebuiltNode::Element(element),
                            name: None,
                        });
                    }
                    if crops_to_clean_aperture {
                        new_chain.push(clean_aperture_crop_item(registry)?);
                    }
                    upstream = None;
                    before_upstream = None;
                }
                Item::Element(spec) => {
                    before_upstream = upstream.replace((spec.name.clone(), spec.props.clone()));
                    new_chain.push(Item::Element(spec));
                }
                Item::Ref { name, pad } => {
                    upstream = None;
                    before_upstream = None;
                    new_chain.push(Item::Ref { name, pad });
                }
                // A pre-built node (from `uridecodebin` / `playbin`) carries no
                // name to take declared caps from, so a `decodebin` cannot follow
                // it. Clear the upstream; if a decodebin does follow, it reports
                // the missing-upstream error.
                prebuilt @ Item::Prebuilt { .. } => {
                    upstream = None;
                    before_upstream = None;
                    new_chain.push(prebuilt);
                }
            }
        }
        out.push(new_chain);
    }
    Ok(out)
}

// the streams of the local file `file` opened, when a container hook can list them
fn demuxed_file_streams(
    registry: &Registry,
    file: &str,
    file_props: &[(String, String)],
) -> Result<Option<(String, ContainerStreams)>, ParseError> {
    let Some((_, location)) = file_props.iter().find(|(key, _)| key == "location") else {
        return Ok(None);
    };
    let container = resolve_upstream_caps(registry, file, file_props)?;
    Ok(registry
        .container_streams(location, &container)
        .map(|streams| (location.clone(), streams)))
}

/// `encodebin`: not an element either, but a macro that expands, at parse time,
/// into one encoder per profile stream plus the muxer that writes the profile's
/// container (M1089). `encodebin2` is gst's name for the same thing.
fn is_encodebin(name: &str) -> bool {
    matches!(name, "encodebin" | "encodebin2")
}

/// The separator between an encoding profile's container and its streams, the
/// same string form gst's `GstEncodingProfile` serializes to.
const PROFILE_SEPARATOR: char = ':';

/// A parsed `profile=`: the container to mux into (`None` for a container-less
/// profile, which is one coded stream and no muxer), and each stream it carries.
struct EncodingProfile {
    container: Option<Caps>,
    streams: Vec<ProfileStream>,
}

/// One stream of a profile: the coded form it names, plus whatever its part
/// pins (M1097). A pinned raw field is honoured by the converter spliced ahead
/// of the encoder; a pinned bitrate is a property on the encoder itself.
struct ProfileStream {
    caps: Caps,
    width: Option<u32>,
    height: Option<u32>,
    /// Frames per second as the `numerator/denominator` `videorate` takes.
    framerate: Option<(u32, u32)>,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    /// Bits per second, gst's `bitrate` unit.
    bitrate: Option<u64>,
}

/// The encoder property a profile's pinned bitrate is applied through.
const BITRATE_PROPERTY: &str = "bitrate";

/// Parse `container-caps:stream-caps[:stream-caps...]`.
///
/// The media type of each part chooses the element: the container names the
/// muxer, a stream names the codec its encoder must produce. A stream part may
/// also pin a width / height / framerate (or a sample `rate` / `channels`),
/// which splices the converter that reaches it, and a `bitrate`, which is set on
/// the encoder. Any other field is refused rather than dropped.
fn parse_encoding_profile(text: &str) -> Result<EncodingProfile, ParseError> {
    let bad = |msg: &str| ParseError::EncodingProfile(msg.to_string());
    let mut parts = text.split(PROFILE_SEPARATOR).map(str::trim).peekable();
    let first = *parts
        .peek()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| bad("empty"))?;
    let head = single_caps(first).ok_or_else(|| {
        ParseError::EncodingProfile(alloc::format!("unknown media type `{first}`"))
    })?;
    // A leading container is the muxing form; a leading stream is gst's
    // container-less profile, which encodes and stops.
    let container = match head {
        Caps::ByteStream { .. } => {
            parts.next();
            Some(head)
        }
        _ => None,
    };
    let mut streams = Vec::new();
    for part in parts {
        if part.is_empty() {
            return Err(bad("empty stream"));
        }
        streams.push(parse_profile_stream(part)?);
    }
    if streams.is_empty() {
        return Err(bad("no stream"));
    }
    Ok(EncodingProfile { container, streams })
}

/// Parse one stream part: its media type plus the fields it pins.
///
/// The caps parser ignores a field it has no home for, so the pinned settings
/// are read from the text here instead: an audio caps carries a channel count
/// and a sample rate whether or not the part named them, and `bitrate` is not a
/// caps field at all.
fn parse_profile_stream(part: &str) -> Result<ProfileStream, ParseError> {
    let caps = single_caps(part)
        .ok_or_else(|| ParseError::EncodingProfile(alloc::format!("unknown stream `{part}`")))?;
    let video = matches!(caps, Caps::CompressedVideo { .. } | Caps::RawVideo { .. });
    let audio = matches!(caps, Caps::Audio { .. });
    let mut stream = ProfileStream {
        caps,
        width: None,
        height: None,
        framerate: None,
        sample_rate: None,
        channels: None,
        bitrate: None,
    };
    for field in crate::caps_parse::split_top_commas(part)
        .into_iter()
        .skip(1)
    {
        let (key, value) = field.split_once('=').ok_or_else(|| {
            ParseError::EncodingProfile(alloc::format!(
                "`{part}` has a field `{}` that is not key=value",
                field.trim()
            ))
        })?;
        let (key, value) = (key.trim(), value.trim());
        match key {
            "width" if video => {
                stream.width = Some(profile_number(part, key, value, MAX_PROFILE_VALUE)? as u32)
            }
            "height" if video => {
                stream.height = Some(profile_number(part, key, value, MAX_PROFILE_VALUE)? as u32)
            }
            "framerate" if video => stream.framerate = Some(profile_framerate(part, value)?),
            "rate" if audio => {
                stream.sample_rate = Some(profile_number(part, key, value, MAX_PROFILE_VALUE)? as u32)
            }
            "channels" if audio => {
                stream.channels = Some(profile_number(part, key, value, MAX_CHANNELS)? as u8)
            }
            BITRATE_PROPERTY => {
                stream.bitrate = Some(profile_number(part, key, value, MAX_PROFILE_VALUE)?)
            }
            _ => {
                return Err(ParseError::EncodingProfile(alloc::format!(
                    "`{part}` pins `{key}`, which a profile cannot apply: set it on an element ahead of the encodebin instead"
                )))
            }
        }
    }
    // The scaler takes a whole output geometry, so half of one has nothing to
    // scale to.
    if stream.width.is_some() != stream.height.is_some() {
        return Err(ParseError::EncodingProfile(alloc::format!(
            "`{part}` pins one of width / height: a scale needs both, so pin both or neither"
        )));
    }
    Ok(stream)
}

/// Upper bounds on a pinned profile field, so a bogus value fails the parse
/// rather than reaching an element as a truncated cast. Every field but the
/// channel count is a `u32` where it lands.
const MAX_PROFILE_VALUE: u64 = u32::MAX as u64;
const MAX_CHANNELS: u64 = u8::MAX as u64;

/// A pinned profile field's number. Zero is rejected everywhere it appears
/// (a zero geometry, rate, channel count or bitrate pins nothing).
fn profile_number(part: &str, key: &str, value: &str, max: u64) -> Result<u64, ParseError> {
    value
        .parse::<u64>()
        .ok()
        .filter(|n| (1..=max).contains(n))
        .ok_or_else(|| {
            ParseError::EncodingProfile(alloc::format!(
                "`{part}` has a bad `{key}={value}`: expected a whole number 1..={max}"
            ))
        })
}

/// A pinned `framerate=N/D` (or a bare `framerate=N`, gst's shorthand), in the
/// fraction form `videorate`'s property takes.
fn profile_framerate(part: &str, value: &str) -> Result<(u32, u32), ParseError> {
    let (numerator, denominator) = value.split_once('/').unwrap_or((value, "1"));
    Ok((
        profile_number(part, "framerate", numerator.trim(), MAX_PROFILE_VALUE)? as u32,
        profile_number(part, "framerate", denominator.trim(), MAX_PROFILE_VALUE)? as u32,
    ))
}

/// One caps from a media-type description. A format-less raw description
/// (`video/x-raw`, `audio/x-raw`) expands to every format it could be, and an
/// uncompressed profile stream means "store what arrives" rather than any one of
/// them, so its first alternative stands for the whole set: the expansion reads
/// only which kind of stream it is. `None` when the text names no media type
/// this build knows.
fn single_caps(desc: &str) -> Option<Caps> {
    let set = CapsSet::from_gst_string(desc)?;
    match set.alternatives() {
        [only] => Some(only.clone()),
        [first, ..] if is_raw_video(first) || is_raw_audio(first) => Some(first.clone()),
        _ => None,
    }
}

/// The profile stream that encodes `input`: a raw video input takes the first
/// coded video stream, raw audio the first coded audio one. `None` when the
/// profile carries nothing of that kind.
fn stream_for_input<'a>(profile: &'a EncodingProfile, input: &Caps) -> Option<&'a ProfileStream> {
    let want_video = is_raw_video(input);
    let want_audio = is_raw_audio(input);
    profile.streams.iter().find(|stream| match stream.caps {
        Caps::CompressedVideo { .. } => want_video,
        Caps::Audio { .. } => want_audio,
        _ => false,
    })
}

/// The converters an encoder needs ahead of it, when the raw stream reaching it
/// is not in a form it takes: a pixel-format convert for video, a sample-format
/// convert plus a resampler for audio (either the format or the rate can be the
/// mismatch). Asked of the encoder itself rather than guessed from a template,
/// so an encoder that already accepts what arrives gets nothing spliced.
///
/// With no upstream caps to check (a decode chain the parser expanded has no
/// name to ask), the converters go in anyway: they pass a matching format
/// through, where a missing one fails the negotiation.
fn conversion_specs(
    registry: &Registry,
    encoder: &ElementSpec,
    input: Option<&Caps>,
    stream: &Caps,
) -> Vec<ElementSpec> {
    let plain = |name: &str| ElementSpec {
        name: name.to_string(),
        props: Vec::new(),
        instance: None,
        log_category: None,
    };
    if let Some(input) = input {
        let accepted = registry
            .make_element(&encoder.name)
            .is_some_and(|element| element.intercept_caps(input).is_ok());
        if accepted {
            return Vec::new();
        }
    }
    // Which converters belong to this stream is the stream's own kind: the
    // encoder for a video stream takes raw video, whatever reaches it.
    match stream {
        Caps::CompressedVideo { .. } => Vec::from([plain("videoconvert")]),
        Caps::Audio { .. } => Vec::from([plain("audioconvert"), plain("audioresample")]),
        _ => Vec::new(),
    }
}

/// The encoder element for one profile stream, as a spec the builder constructs
/// like any named element. The properties come from the registry's choice (a
/// multi-codec encoder is pinned to this codec with `codec=`), plus the
/// profile's own `bitrate` where it pinned one.
fn encoder_spec(
    registry: &Registry,
    stream: &ProfileStream,
) -> Result<Option<ElementSpec>, ParseError> {
    // An uncompressed stream profile (`audio/x-raw` in a WAV, `video/x-raw` in a
    // y4m) stores what arrives, so there is nothing to encode.
    if is_raw_video(&stream.caps) || is_raw_audio(&stream.caps) {
        if let Some(bitrate) = stream.bitrate {
            return Err(ParseError::EncodingProfile(alloc::format!(
                "pins `bitrate={bitrate}` on an uncompressed stream, which has no encoder to set it on"
            )));
        }
        return Ok(None);
    }
    let choice = registry.encoder_choice(&stream.caps).ok_or_else(|| {
        ParseError::NoEncoder(alloc::format!(
            "no encoder for {} is compiled in (build the feature for it, or name an encoder explicitly)",
            stream.caps.to_gst_string()
        ))
    })?;
    let mut props: Vec<(String, String)> = choice
        .props
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    if let Some(bitrate) = stream.bitrate {
        if !declares_bitrate(registry, choice.element) {
            return Err(ParseError::EncodingProfile(alloc::format!(
                "pins `bitrate={bitrate}`, which `{}` has no property for: drop it, or name the encoder yourself with the knob it does have",
                choice.element
            )));
        }
        props.push((BITRATE_PROPERTY.to_string(), alloc::format!("{bitrate}")));
    }
    Ok(Some(ElementSpec {
        name: choice.element.to_string(),
        props,
        instance: None,
        log_category: None,
    }))
}

/// Whether an encoder takes a target bitrate at all, asked of the element rather
/// than assumed, so a profile that pins one on a fixed-rate encoder fails the
/// parse instead of encoding at some other rate.
fn declares_bitrate(registry: &Registry, element: &str) -> bool {
    registry.make_element(element).is_some_and(|element| {
        element
            .properties()
            .iter()
            .any(|spec| spec.name == BITRATE_PROPERTY)
    })
}

/// The elements one branch of a profile expands to: the converters its encoder
/// needs, the converters its pinned settings need, then the encoder itself. A
/// pinned setting on an uncompressed stream still gets its converter, there is
/// just nothing after it but the muxer.
fn stream_specs(
    registry: &Registry,
    stream: &ProfileStream,
    upstream: Option<&(String, Vec<(String, String)>)>,
) -> Result<Vec<ElementSpec>, ParseError> {
    let encoder = encoder_spec(registry, stream)?;
    let mut specs = match &encoder {
        Some(encoder) => {
            let input = upstream
                .and_then(|(name, props)| resolve_upstream_caps(registry, name, props).ok());
            conversion_specs(registry, encoder, input.as_ref(), &stream.caps)
        }
        None => Vec::new(),
    };
    splice_pinned_settings(&mut specs, stream);
    specs.extend(encoder);
    Ok(specs)
}

/// Apply a stream's pinned settings to the converters ahead of its encoder: a
/// geometry sets `videoscale`, a framerate `videorate`, a channel count
/// `audioconvert` and a sample rate `audioresample`, adding that converter when
/// the encoder did not already need one.
fn splice_pinned_settings(specs: &mut Vec<ElementSpec>, stream: &ProfileStream) {
    if let (Some(width), Some(height)) = (stream.width, stream.height) {
        pin_on_converter(specs, "videoscale", "width", alloc::format!("{width}"));
        pin_on_converter(specs, "videoscale", "height", alloc::format!("{height}"));
    }
    if let Some((numerator, denominator)) = stream.framerate {
        pin_on_converter(
            specs,
            "videorate",
            "framerate",
            alloc::format!("{numerator}/{denominator}"),
        );
    }
    if let Some(channels) = stream.channels {
        pin_on_converter(
            specs,
            "audioconvert",
            "channels",
            alloc::format!("{channels}"),
        );
    }
    if let Some(sample_rate) = stream.sample_rate {
        pin_on_converter(
            specs,
            "audioresample",
            "samplerate",
            alloc::format!("{sample_rate}"),
        );
    }
}

/// Set a property on the named converter, appending the converter when the list
/// does not already carry it.
fn pin_on_converter(specs: &mut Vec<ElementSpec>, element: &str, key: &str, value: String) {
    match specs.iter_mut().find(|spec| spec.name == element) {
        Some(spec) => spec.props.push((key.to_string(), value)),
        None => specs.push(ElementSpec {
            name: element.to_string(),
            props: Vec::from([(key.to_string(), value)]),
            instance: None,
            log_category: None,
        }),
    }
}

/// The stream index a `togglerecord` request pad names, and which side it is.
/// `None` for a pad that is not a request pad at all.
fn request_pad(pad: &str) -> Option<(bool, usize)> {
    for (prefix, is_sink) in [(TOGGLERECORD_SINK_PAD, true), (TOGGLERECORD_SRC_PAD, false)] {
        if let Some(index) = pad.strip_prefix(prefix) {
            return index.parse().ok().map(|i| (is_sink, i));
        }
    }
    None
}

/// The instance names of the `togglerecord` keywords a line declared, in
/// appearance order. An unnamed one cannot be referenced, so it stays the
/// single-stream element it already is.
fn named_togglerecords(chains: &[Chain]) -> Vec<String> {
    let mut names = Vec::new();
    for chain in chains {
        for item in chain {
            if let Item::Element(spec) = item {
                if spec.name == TOGGLERECORD {
                    if let Some(instance) = &spec.instance {
                        names.push(instance.clone());
                    }
                }
            }
        }
    }
    names
}

/// The name the expansion gives the element carrying stream `index` of the
/// `togglerecord` named `base`. Stream 0 is the inline keyword itself, which
/// keeps the name the line gave it.
fn togglerecord_stream_name(base: &str, index: usize) -> String {
    alloc::format!("{base}-{index}")
}

/// One secondary stream's element: the same keyword in the same group, not the
/// main stream.
fn togglerecord_secondary(
    base: &str,
    group: &str,
    index: usize,
    log_category: Option<&str>,
) -> ElementSpec {
    ElementSpec {
        name: TOGGLERECORD.to_string(),
        props: alloc::vec![
            (TOGGLERECORD_GROUP_PROP.to_string(), group.to_string()),
            (TOGGLERECORD_MAIN_PROP.to_string(), "false".to_string()),
        ],
        instance: Some(togglerecord_stream_name(base, index)),
        log_category: log_category.map(|c| c.to_string()),
    }
}

/// `togglerecord name=t` with request pads (M1172): gst spells several streams
/// that start and stop together as one element with `sink_%u` / `src_%u` pad
/// pairs. Here that is one `ToggleRecord` per stream sharing one group, built by
/// rewriting the references, so the graph holds plain 1-in 1-out transforms the
/// way every other g2g bin flattens.
///
/// The inline keyword is stream 0, the main stream whose keyframes decide. Each
/// `t.sink_K` tail reference becomes a secondary element in that chain, and the
/// matching `t.src_K` head reference becomes a reference to it. The group is the
/// keyword's own `name=` unless the line set `group=` itself.
///
/// A reference with no partner, or one naming no request pad, is a parse error:
/// the pads are paired, so an unpaired one would leave a stream with one end
/// dangling.
fn expand_togglerecord(chains: Vec<Chain>) -> Result<Vec<Chain>, ParseError> {
    let bases = named_togglerecords(&chains);
    if bases.is_empty() {
        return Ok(chains);
    }
    let mut chains = chains;
    for base in bases {
        // What the group is called, and the spec the secondaries copy their log
        // category from.
        let mut group = base.clone();
        // The keyword's own log category, which its secondaries inherit.
        let mut log_category: Option<String> = None;
        let mut declared = false;
        for chain in &chains {
            for item in chain {
                if let Item::Element(spec) = item {
                    if spec.name == TOGGLERECORD && spec.instance.as_deref() == Some(&base) {
                        if let Some((_, value)) = spec
                            .props
                            .iter()
                            .find(|(k, _)| k == TOGGLERECORD_GROUP_PROP)
                        {
                            group = value.clone();
                        }
                        log_category = spec.log_category.clone();
                        declared = true;
                    }
                }
            }
        }
        if !declared {
            continue;
        }

        // Every request pad the line references, checked in pairs before
        // anything is rewritten.
        let (mut sinks, mut srcs): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for chain in &chains {
            for item in chain {
                let Item::Ref { name, pad } = item else {
                    continue;
                };
                if name != &base {
                    continue;
                }
                match request_pad(pad) {
                    Some((true, index)) if index > 0 => sinks.push(index),
                    Some((false, index)) if index > 0 => srcs.push(index),
                    _ => {
                        return Err(ParseError::BadRequestPad {
                            element: base.clone(),
                            pad: pad.clone(),
                        })
                    }
                }
            }
        }
        for (side, other) in [(&sinks, &srcs), (&srcs, &sinks)] {
            if let Some(index) = side.iter().find(|i| !other.contains(i)) {
                let pad = match core::ptr::eq(side, &sinks) {
                    true => alloc::format!("{TOGGLERECORD_SINK_PAD}{index}"),
                    false => alloc::format!("{TOGGLERECORD_SRC_PAD}{index}"),
                };
                return Err(ParseError::UnpairedRequestPad {
                    element: base.clone(),
                    pad,
                });
            }
        }
        if sinks.is_empty() {
            continue; // a named `togglerecord` nobody referenced: one stream
        }

        // Rewrite: the inline keyword joins the group, each `sink_K` becomes the
        // secondary element itself, and each `src_K` becomes a reference to it.
        for chain in &mut chains {
            for item in chain.iter_mut() {
                match item {
                    Item::Element(spec)
                        if spec.name == TOGGLERECORD && spec.instance.as_deref() == Some(&base) =>
                    {
                        if !spec.props.iter().any(|(k, _)| k == TOGGLERECORD_GROUP_PROP) {
                            spec.props
                                .push((TOGGLERECORD_GROUP_PROP.to_string(), group.clone()));
                        }
                    }
                    Item::Ref { name, pad } if name == &base => {
                        // Checked above, so the pad is a request pad above zero.
                        let (is_sink, index) = request_pad(pad).expect("checked request pad");
                        *item = match is_sink {
                            true => Item::Element(togglerecord_secondary(
                                &base,
                                &group,
                                index,
                                log_category.as_deref(),
                            )),
                            false => Item::Ref {
                                name: togglerecord_stream_name(&base, index),
                                pad: String::new(),
                            },
                        };
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(chains)
}

/// `transcodebin`: decode whatever arrives and re-encode it to a profile, which
/// is exactly `decodebin ! encodebin profile=...`. Rewritten into those two
/// macros before either expands, so it inherits both behaviours (and both error
/// messages) rather than repeating them.
fn expand_transcodebin(chains: Vec<Chain>) -> Vec<Chain> {
    let mut out = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut new_chain: Chain = Vec::with_capacity(chain.len() + 1);
        for item in chain {
            match item {
                Item::Element(spec) if spec.name == "transcodebin" => {
                    new_chain.push(Item::Element(ElementSpec {
                        name: "decodebin".to_string(),
                        props: Vec::new(),
                        instance: None,
                        log_category: spec.log_category.clone(),
                    }));
                    new_chain.push(Item::Element(ElementSpec {
                        name: "encodebin".to_string(),
                        ..spec
                    }));
                }
                other => new_chain.push(other),
            }
        }
        out.push(new_chain);
    }
    out
}

/// Expand every `encodebin` into `<encoder> ! <muxer>`: the encoder for the
/// stream matching that chain's input, and the muxer for the profile's container.
/// The muxer inherits the bin's `name=`, so a second branch reaching it as `e.`
/// links into the muxer the way any fan-in does; each such branch gets the
/// encoder for its own kind of input inserted ahead of the reference.
fn expand_encodebin(registry: &Registry, chains: Vec<Chain>) -> Result<Vec<Chain>, ParseError> {
    // Nothing to do for the common line with no encodebin in it.
    if !chains
        .iter()
        .flatten()
        .any(|item| matches!(item, Item::Element(spec) if is_encodebin(&spec.name)))
    {
        return Ok(chains);
    }

    // Every encodebin's profile and instance name, so a branch referencing the
    // name knows which profile to encode for.
    let mut profiles: Vec<(Option<String>, EncodingProfile)> = Vec::new();
    for chain in &chains {
        for item in chain {
            if let Item::Element(spec) = item {
                if is_encodebin(&spec.name) {
                    let text = prop(spec, "profile")
                        .ok_or_else(|| ParseError::EncodingProfile("missing".to_string()))?;
                    for (key, _) in &spec.props {
                        if !matches!(key.as_str(), "profile") {
                            return Err(ParseError::EncodingProfile(alloc::format!(
                                "unknown property `{key}`"
                            )));
                        }
                    }
                    profiles.push((spec.instance.clone(), parse_encoding_profile(text)?));
                }
            }
        }
    }

    let mut out = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut new_chain: Chain = Vec::with_capacity(chain.len() + 1);
        let mut upstream: Option<(String, Vec<(String, String)>)> = None;
        for item in chain {
            match item {
                Item::Element(spec) if is_encodebin(&spec.name) => {
                    let profile = profiles
                        .iter()
                        .find(|(instance, _)| *instance == spec.instance)
                        .map(|(_, profile)| profile)
                        .expect("every encodebin's profile was parsed above");
                    // The container is the profile's head, so a missing muxer is
                    // reported before a missing encoder.
                    let muxer = muxer_spec(registry, profile, &spec)?;
                    let stream = profile_stream(registry, profile, upstream.as_ref())?;
                    for element in stream_specs(registry, stream, upstream.as_ref())? {
                        new_chain.push(Item::Element(element));
                    }
                    if let Some(muxer) = muxer {
                        new_chain.push(Item::Element(muxer));
                    }
                    upstream = None;
                }
                // A branch feeding an encodebin by name encodes its own input
                // first: the reference links into the muxer, which takes coded
                // streams.
                Item::Ref { name, pad } => {
                    if let Some((_, profile)) = profiles
                        .iter()
                        .find(|(instance, _)| instance.as_deref() == Some(name.as_str()))
                    {
                        let stream = profile_stream(registry, profile, upstream.as_ref())?;
                        for element in stream_specs(registry, stream, upstream.as_ref())? {
                            new_chain.push(Item::Element(element));
                        }
                    }
                    upstream = None;
                    new_chain.push(Item::Ref { name, pad });
                }
                Item::Element(spec) => {
                    upstream = Some((spec.name.clone(), spec.props.clone()));
                    new_chain.push(Item::Element(spec));
                }
                prebuilt @ Item::Prebuilt { .. } => {
                    upstream = None;
                    new_chain.push(prebuilt);
                }
            }
        }
        out.push(new_chain);
    }
    Ok(out)
}

/// The profile stream a branch encodes into. A single-stream profile leaves no
/// choice, so it needs no input caps at all (which is what lets `decodebin !
/// encodebin` work: an expanded decode chain has no name to ask for caps). With
/// several streams the branch's own input decides which one, so the element
/// ahead of it has to declare its output.
fn profile_stream<'a>(
    registry: &Registry,
    profile: &'a EncodingProfile,
    upstream: Option<&(String, Vec<(String, String)>)>,
) -> Result<&'a ProfileStream, ParseError> {
    if let [only] = &profile.streams[..] {
        return Ok(only);
    }
    let input = encodebin_input_caps(registry, upstream)?;
    stream_for_input(profile, &input).ok_or_else(|| {
        ParseError::EncodebinInput(alloc::format!(
            "the profile carries no stream that encodes {}",
            input.to_gst_string()
        ))
    })
}

/// The raw caps reaching an `encodebin` (or a branch that references one): the
/// declared output of the element ahead of it.
fn encodebin_input_caps(
    registry: &Registry,
    upstream: Option<&(String, Vec<(String, String)>)>,
) -> Result<Caps, ParseError> {
    let (name, props) = upstream.ok_or_else(|| {
        ParseError::EncodebinInput(
            "no upstream element to encode; it must follow a source or element with declared caps"
                .to_string(),
        )
    })?;
    resolve_upstream_caps(registry, name, props).map_err(|_| {
        ParseError::EncodebinInput(alloc::format!("`{name}` declares no output caps to encode"))
    })
}

/// The muxer element for a profile's container, carrying the bin's instance name
/// so a `name.` reference resolves to it.
fn muxer_spec(
    registry: &Registry,
    profile: &EncodingProfile,
    bin: &ElementSpec,
) -> Result<Option<ElementSpec>, ParseError> {
    let Some(container) = &profile.container else {
        return Ok(None);
    };
    let name = registry.muxer_name(container).ok_or_else(|| {
        ParseError::NoEncoder(alloc::format!(
            "no muxer for {} is compiled in",
            container.to_gst_string()
        ))
    })?;
    Ok(Some(ElementSpec {
        name: name.to_string(),
        props: Vec::new(),
        instance: bin.instance.clone(),
        log_category: bin.log_category.clone(),
    }))
}

/// The caps a `decodebin` predecessor produces, used as the auto-plug input
/// (M195). For a registered source, build it and apply its properties so a
/// property that re-types the output (a `filesrc`'s `bytestream-format`) is
/// reflected via [`SourceLoop::probe_output_caps`]; fall back to the registry's
/// declared caps (a fixed source, or a transform's source-pad template).
/// `probe_output_caps` also sniffs a `bytestream-format=auto` source's header at
/// parse time (M480), so `decodebin` picks the demuxer from the real content even
/// when the file extension is wrong, and content it does not recognize fails the
/// parse.
fn resolve_upstream_caps(
    registry: &Registry,
    name: &str,
    props: &[(String, String)],
) -> Result<Caps, ParseError> {
    if let Some(mut src) = registry.make_source(name) {
        apply_props(&mut src, name, props)?;
        // `probe_output_caps` may sniff the header (a `bytestream-format=auto`
        // source), so `decodebin` picks the demuxer from the real content, not a
        // mislabeled extension; it falls back to the no-I/O caps otherwise.
        let probed = src
            .probe_output_caps()
            .map_err(|_| ParseError::UnrecognizedContent(name.to_string()))?;
        if let Some(caps) = probed {
            return Ok(caps);
        }
    }
    registry
        .declared_output_caps(name)
        .ok_or(ParseError::DecodebinNoUpstream)
}

/// `uridecodebin` / `playbin` / `fallbacksrc`: a source-providing macro.
/// `uridecodebin uri=X` builds the source from the URI scheme handler and
/// auto-plugs the decode chain to raw; `playbin uri=X` is that plus an auto sink
/// (`autovideosink`, or the `video-sink=` override), i.e. a complete pipeline;
/// `fallbacksrc uri=X` is `uridecodebin` plus a second branch, both feeding a
/// `fallbackswitch`.
fn is_uri_source(name: &str) -> bool {
    matches!(name, "uridecodebin" | "playbin" | "fallbacksrc")
}

/// The value of a spec property by key, if present.
fn prop<'a>(spec: &'a ElementSpec, key: &str) -> Option<&'a str> {
    spec.props
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// The properties the `fallbacksrc` keyword takes. It is a macro, not an element,
/// so the expansion reads these itself and an unknown one is rejected here
/// instead of by `apply_props`.
const FALLBACKSRC_PROPS: &[&str] = &[
    "uri",
    "fallback-uri",
    "timeout",
    "immediate-fallback",
    "enable-audio",
    "enable-video",
    "restart-on-eos",
    "restart-timeout",
    "retry-timeout",
    "manual-unblock",
];

/// gst `fallbacksrc`'s defaults for its restart timeouts (M1163): a source that
/// delivers nothing for `restart-timeout` is rebuilt, and rebuilding is given up
/// after `retry-timeout` of failures.
const FALLBACK_RESTART_TIMEOUT_NS: u64 = 5_000_000_000;
const FALLBACK_RETRY_TIMEOUT_NS: u64 = 60_000_000_000;

/// The launch keyword whose `sink_%u` / `src_%u` request pads expand into one
/// element per stream, all joined by a shared group (M1172).
const TOGGLERECORD: &str = "togglerecord";

/// The properties the expansion writes on the elements it builds: the group that
/// joins them, and which of them is the stream whose keyframes decide.
const TOGGLERECORD_GROUP_PROP: &str = "group";
const TOGGLERECORD_MAIN_PROP: &str = "main";

/// Request-pad prefixes on a `togglerecord` reference: the input a stream enters
/// by and the output it leaves by.
const TOGGLERECORD_SINK_PAD: &str = "sink_";
const TOGGLERECORD_SRC_PAD: &str = "src_";

/// The switch a `fallbacksrc` expands around, and the input its fallback branch
/// links to (input 0 is the main URI, so the fallback is input 1).
const FALLBACK_SWITCH: &str = "fallbackswitch";
const FALLBACK_SWITCH_PAD: &str = "sink_1";
const FALLBACK_SWITCH_INPUTS: usize = 2;

/// The instance name given to the generated switch when the `fallbacksrc` had no
/// `name=` of its own, so the fallback branch has something to reference. A
/// launch line using this name for one of its own elements collides
/// ([`ParseError::DuplicateName`]).
pub const FALLBACK_SWITCH_NAME: &str = "g2g-fallbacksrc-switch";

/// Neither test source is live, so the dummy fallback would run as fast as the
/// CPU allows; `clocksync` holds each buffer until its PTS comes due, which is
/// what makes the dummy play at the main stream's rate.
const FALLBACK_PACER: &str = "clocksync";

/// Instance-name suffixes for the two URI sources the expansion builds (M1164),
/// on the `fallbacksrc`'s own name: `fallbacksrc name=fb` gives `fb-source` and
/// `fb-fallback-source`, so a launch line can address either.
pub const FALLBACK_MAIN_SOURCE_SUFFIX: &str = "-source";
pub const FALLBACK_FALLBACK_SOURCE_SUFFIX: &str = "-fallback-source";

/// Instance-name suffixes telling the per-kind switches of a fanned-out
/// `fallbacksrc` apart (M1168): `fallbacksrc name=fb` over a file carrying both
/// kinds gives the switches `fb-video` and `fb-audio`.
pub const FALLBACK_VIDEO_SUFFIX: &str = "-video";
pub const FALLBACK_AUDIO_SUFFIX: &str = "-audio";

/// The dummy fallback source per stream kind: black frames, or silence.
const FALLBACK_VIDEO_DUMMY: (&str, &str, &str) = ("videotestsrc", "pattern", "black");
const FALLBACK_AUDIO_DUMMY: (&str, &str, &str) = ("audiotestsrc", "wave", "silence");

/// The sink a `fallbacksrc` that is the whole pipeline ends each kind on, so a
/// lone keyword is a runnable line rather than a switch with nothing downstream.
const FALLBACK_VIDEO_SINK: &str = "autovideosink";
const FALLBACK_AUDIO_SINK: &str = "autoaudiosink";

/// Between a raw-audio switch output and its sink: the sink fixes one PCM format
/// at configure, and these absorb the stream's real channels and rate.
const FALLBACK_AUDIO_TAIL: &[&str] = &["audioconvert", "audioresample"];

/// What one stream kind's branch of a fanned-out `fallbacksrc` is made of: the
/// suffix its switch is named with, the generator its dummy fallback uses, the
/// elements between the switch and the sink, the sink itself, and the raw caps
/// the decode chain plugs to.
#[derive(Clone, Copy)]
struct FallbackKind {
    suffix: &'static str,
    dummy: (&'static str, &'static str, &'static str),
    tail: &'static [&'static str],
    sink: &'static str,
    raw: fn(&Caps) -> bool,
}

/// The branch shape for a demux port's media kind, or `None` for a kind a
/// `fallbacksrc` has no generator and no sink for (text, unclassified).
fn fallback_kind(stream_type: StreamType) -> Option<FallbackKind> {
    match stream_type {
        StreamType::Video => Some(FallbackKind {
            suffix: FALLBACK_VIDEO_SUFFIX,
            dummy: FALLBACK_VIDEO_DUMMY,
            tail: &[],
            sink: FALLBACK_VIDEO_SINK,
            raw: is_raw_video,
        }),
        StreamType::Audio => Some(FallbackKind {
            suffix: FALLBACK_AUDIO_SUFFIX,
            dummy: FALLBACK_AUDIO_DUMMY,
            tail: FALLBACK_AUDIO_TAIL,
            sink: FALLBACK_AUDIO_SINK,
            raw: is_raw_audio,
        }),
        _ => None,
    }
}

/// A `fallbacksrc` keyword's boolean property, defaulting to `default` when the
/// line did not write it. Spelled as gst writes booleans in a launch line.
fn keyword_bool(spec: &ElementSpec, key: &str, default: bool) -> Result<bool, ParseError> {
    let Some(text) = prop(spec, key) else {
        return Ok(default);
    };
    match text {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(ParseError::BadValue {
            element: spec.name.clone(),
            key: key.to_string(),
            value: text.to_string(),
        }),
    }
}

/// A `fallbacksrc` keyword's unsigned property (a nanosecond timeout), defaulting
/// to `default` when the line did not write it.
fn keyword_uint(spec: &ElementSpec, key: &str, default: u64) -> Result<u64, ParseError> {
    let Some(text) = prop(spec, key) else {
        return Ok(default);
    };
    text.parse().map_err(|_| ParseError::BadValue {
        element: spec.name.clone(),
        key: key.to_string(),
        value: text.to_string(),
    })
}

/// Wrap a `fallbacksrc` URI source so it is rebuilt when it dies (M1163), when
/// the registry has a restart hook; else the source runs as it is. `unblock` is
/// the handle each life waits on under `manual-unblock=true` (M1166).
fn restartable_uri_source(
    registry: &Registry,
    source: Box<dyn DynSourceLoop>,
    uri: &str,
    policy: RestartPolicy,
    role: FallbackSourceRole,
    unblock: Option<UnblockHandle>,
) -> Result<Box<dyn DynSourceLoop>, ParseError> {
    if registry.restart_source_hook().is_none() {
        return Ok(source);
    }
    let rebuild = registry
        .uri_source_rebuilder(uri)
        .map_err(|e: UriError| ParseError::Uri(alloc::format!("{uri}: {e:?}")))?;
    Ok(restartable_source(
        registry, source, rebuild, policy, role, unblock,
    ))
}

/// [`restartable_uri_source`] with the rebuild already in hand: the fan-out path
/// rebuilds the container's byte source, not the URI scheme handler's (which for
/// `file://` may self-demux a different container).
fn restartable_source(
    registry: &Registry,
    source: Box<dyn DynSourceLoop>,
    rebuild: UriRebuild,
    policy: RestartPolicy,
    role: FallbackSourceRole,
    unblock: Option<UnblockHandle>,
) -> Box<dyn DynSourceLoop> {
    match registry.restart_source_hook() {
        Some(hook) => hook(source, rebuild, policy, role, unblock),
        None => source,
    }
}

/// [`restartable_source`] for a session that produces every port itself (M1170).
fn restartable_fanout_source(
    registry: &Registry,
    source: Box<dyn DynMultiOutputSource>,
    rebuild: FanoutRebuild,
    policy: RestartPolicy,
    role: FallbackSourceRole,
    unblock: Option<UnblockHandle>,
) -> Box<dyn DynMultiOutputSource> {
    match registry.restart_fanout_source_hook() {
        Some(hook) => hook(source, rebuild, policy, role, unblock),
        None => source,
    }
}

enum FallbacksrcSourceOrigin<'a> {
    Uri(&'a str),
    Factory(FallbacksrcSourceFactory),
}

impl<'a> FallbacksrcSourceOrigin<'a> {
    // the URI wins over the factory, as gst's `uri` wins over `source`
    fn resolve(uri: Option<&'a str>, factory: Option<FallbacksrcSourceFactory>) -> Option<Self> {
        uri.map(Self::Uri).or_else(|| factory.map(Self::Factory))
    }

    fn build(
        &self,
        registry: &Registry,
        policy: RestartPolicy,
        role: FallbackSourceRole,
        unblock: Option<UnblockHandle>,
    ) -> Result<UriSource, ParseError> {
        match self {
            Self::Uri(uri) => {
                let built = registry
                    .build_uri_source(uri)
                    .map_err(|e: UriError| ParseError::Uri(alloc::format!("{uri}: {e:?}")))?;
                Ok(UriSource {
                    source: restartable_uri_source(
                        registry,
                        built.source,
                        uri,
                        policy,
                        role,
                        unblock,
                    )?,
                    ..built
                })
            }
            Self::Factory(factory) => {
                let (source, caps) = factory.build();
                Ok(UriSource {
                    source: restartable_source(
                        registry,
                        source,
                        factory.rebuilder(),
                        policy,
                        role,
                        unblock,
                    ),
                    caps,
                    has_clean_aperture: false,
                })
            }
        }
    }
}

/// Everything a `fallbacksrc` keyword's properties say, read once so the
/// single-stream expansion and the lone-keyword fan-out agree on them.
struct FallbackSettings<'a> {
    main: FallbacksrcSourceOrigin<'a>,
    // None means the dummy generators
    fallback: Option<FallbacksrcSourceOrigin<'a>>,
    enable_video: bool,
    enable_audio: bool,
    restart: RestartPolicy,
    unblock: Option<UnblockHandle>,
    /// The properties forwarded to each `fallbackswitch` the expansion builds.
    switch_props: Vec<(String, String)>,
}

/// Read a `fallbacksrc` keyword's properties. It is a macro, not an element, so
/// an unknown property is rejected here instead of by `apply_props`.
fn fallbacksrc_settings<'a>(
    registry: &Registry,
    spec: &'a ElementSpec,
) -> Result<FallbackSettings<'a>, ParseError> {
    if let Some((key, _)) = spec
        .props
        .iter()
        .find(|(k, _)| !FALLBACKSRC_PROPS.contains(&k.as_str()))
    {
        return Err(ParseError::UnknownProperty {
            element: spec.name.clone(),
            key: key.clone(),
        });
    }
    let main =
        FallbacksrcSourceOrigin::resolve(prop(spec, "uri"), registry.fallbacksrc_main_source())
            .ok_or_else(|| ParseError::MissingUri(spec.name.clone()))?;
    let enable_video = keyword_bool(spec, "enable-video", true)?;
    let enable_audio = keyword_bool(spec, "enable-audio", true)?;
    if !enable_video && !enable_audio {
        return Err(ParseError::BadValue {
            element: spec.name.clone(),
            key: "enable-video".to_string(),
            value: "false".to_string(),
        });
    }
    // Without a registered handle a held source could never be released, so the
    // line is rejected here rather than hanging with nothing to poke.
    let unblock = match keyword_bool(spec, "manual-unblock", false)? {
        false => None,
        true => Some(
            registry
                .unblock_handle()
                .ok_or_else(|| ParseError::MissingUnblockHandle(spec.name.clone()))?,
        ),
    };
    let mut switch_props = Vec::new();
    for key in ["timeout", "immediate-fallback"] {
        if let Some(value) = prop(spec, key) {
            switch_props.push((key.to_string(), value.to_string()));
        }
    }
    Ok(FallbackSettings {
        main,
        fallback: FallbacksrcSourceOrigin::resolve(
            prop(spec, "fallback-uri"),
            registry.fallbacksrc_fallback_source(),
        ),
        enable_video,
        enable_audio,
        restart: RestartPolicy {
            restart_on_eos: keyword_bool(spec, "restart-on-eos", false)?,
            restart_timeout_ns: keyword_uint(spec, "restart-timeout", FALLBACK_RESTART_TIMEOUT_NS)?,
            retry_timeout_ns: keyword_uint(spec, "retry-timeout", FALLBACK_RETRY_TIMEOUT_NS)?,
        },
        unblock,
        switch_props,
    })
}

/// Expand one `fallbacksrc` into its main branch (the `uri=` source auto-plugged
/// to raw), the `fallbackswitch` those branches meet at, and the fallback branch
/// as a chain of its own that references the switch's second input.
///
/// The expansion is single-stream, like `uridecodebin`: `enable-video` /
/// `enable-audio` say which kind the decode chain may reach, video first.
/// `consumer` is the element the switch output feeds, whose declared input memory
/// picks the decoder (M1018). `auto_sink` appends the kind's automatic sink, so a
/// `fallbacksrc` that is the whole line is still a runnable pipeline.
fn expand_fallbacksrc(
    registry: &Registry,
    spec: &ElementSpec,
    consumer: Option<&str>,
    switch_name: &str,
    avoided: &[&str],
    auto_sink: bool,
) -> Result<(Vec<Item>, Chain), ParseError> {
    let settings = fallbacksrc_settings(registry, spec)?;
    let FallbackSettings {
        main,
        fallback: fallback_source,
        enable_video,
        enable_audio,
        restart,
        unblock,
        switch_props,
    } = settings;
    let preferred = consumer.map_or(MemoryDomainKind::System, |name| {
        registry.declared_memory_preference(name)
    });
    let main_source = main.build(registry, restart, FallbackSourceRole::Main, unblock.clone())?;
    let source_caps = &main_source.caps;
    // The kind is decided by which target the search reaches, so the dummy
    // fallback is the right generator and a fallback source decodes to the same
    // shape. Video first: a container carrying both is a video stream here.
    let video_target: &dyn Fn(&Caps) -> bool = &is_raw_video;
    let audio_target: &dyn Fn(&Caps) -> bool = &is_raw_audio;
    let plug = |caps: &Caps, target: &dyn Fn(&Caps) -> bool| {
        registry.autoplug_avoiding(caps, target, DECODEBIN_MAX_DEPTH, preferred, avoided)
    };
    let video = enable_video
        .then(|| plug(source_caps, video_target))
        .flatten();
    let (decoders, target, kind) = match video {
        Some(decoders) => (
            decoders,
            video_target,
            fallback_kind(StreamType::Video).expect("video is a fallbacksrc kind"),
        ),
        None => {
            let decoders = enable_audio
                .then(|| plug(source_caps, audio_target))
                .flatten()
                .ok_or_else(|| ParseError::no_decode_chain(source_caps))?;
            (
                decoders,
                audio_target,
                fallback_kind(StreamType::Audio).expect("audio is a fallbacksrc kind"),
            )
        }
    };

    let mut main = Vec::with_capacity(decoders.len() + 2);
    main.push(Item::Prebuilt {
        node: PrebuiltNode::Source(main_source.source),
        name: Some(alloc::format!("{switch_name}{FALLBACK_MAIN_SOURCE_SUFFIX}")),
    });
    for decoder in decoders {
        main.push(Item::Prebuilt {
            node: PrebuiltNode::Element(decoder),
            name: None,
        });
    }
    if main_source.has_clean_aperture {
        main.push(clean_aperture_crop_item(registry)?);
    }
    main.push(Item::Element(ElementSpec {
        name: FALLBACK_SWITCH.to_string(),
        props: switch_props,
        instance: Some(switch_name.to_string()),
        log_category: spec.log_category.clone(),
    }));
    if auto_sink {
        main.push(Item::Element(ElementSpec {
            name: kind.sink.to_string(),
            props: Vec::new(),
            instance: None,
            log_category: None,
        }));
    }

    let mut fallback: Chain = Vec::new();
    match fallback_source {
        Some(origin) => {
            let built = origin.build(registry, restart, FallbackSourceRole::Fallback, unblock)?;
            let decoders = plug(&built.caps, target)
                .ok_or_else(|| ParseError::no_decode_chain(&built.caps))?;
            fallback.push(Item::Prebuilt {
                node: PrebuiltNode::Source(built.source),
                name: Some(alloc::format!(
                    "{switch_name}{FALLBACK_FALLBACK_SOURCE_SUFFIX}"
                )),
            });
            for decoder in decoders {
                fallback.push(Item::Prebuilt {
                    node: PrebuiltNode::Element(decoder),
                    name: None,
                });
            }
            if built.has_clean_aperture {
                fallback.push(clean_aperture_crop_item(registry)?);
            }
        }
        None => {
            let (name, key, value) = kind.dummy;
            fallback.push(Item::Element(ElementSpec {
                name: name.to_string(),
                props: Vec::from([(key.to_string(), value.to_string())]),
                instance: None,
                log_category: None,
            }));
            fallback.push(Item::Element(ElementSpec {
                name: FALLBACK_PACER.to_string(),
                props: Vec::new(),
                instance: None,
                log_category: None,
            }));
        }
    }
    fallback.push(Item::Ref {
        name: switch_name.to_string(),
        pad: FALLBACK_SWITCH_PAD.to_string(),
    });
    Ok((main, fallback))
}

/// Expand every `uridecodebin` / `playbin` / `fallbacksrc` (a source position
/// element) into the `uri=` scheme handler's source plus the auto-plugged decode
/// chain, as pre-built nodes spliced straight into the chain. `playbin`
/// additionally appends an auto sink so the line is a complete pipeline;
/// `fallbacksrc` appends its switch and contributes the fallback branch as
/// another chain. The element must head its chain (it provides the source).
fn expand_uri_sources(
    registry: &Registry,
    chains: Vec<Chain>,
    avoided: &[&str],
) -> Result<Vec<Chain>, ParseError> {
    let mut out = Vec::with_capacity(chains.len());
    // Numbers the generated switch names apart, so two `fallbacksrc` in one line
    // do not collide on the default name.
    let mut fallbacksrc_count = 0usize;
    // A `fallbacksrc` that is the whole line has nothing downstream of its
    // switch, so the expansion appends the sink for the kind it reached.
    let lone_fallbacksrc = lone_fallbacksrc_spec(&chains).is_some();
    let referenced = referenced_names(&chains);
    for chain in chains {
        let mut new_chain: Chain = Vec::with_capacity(chain.len());
        let mut fallback_chains: Vec<Chain> = Vec::new();
        let mut items = chain.into_iter().enumerate().peekable();
        while let Some((i, item)) = items.next() {
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
            if spec.name == "fallbacksrc" {
                let consumer = match items.peek() {
                    Some((_, Item::Element(consumer))) => Some(consumer.name.as_str()),
                    _ => None,
                };
                // A `fallbacksrc name=f` names its switch, so `f.` resolves to
                // the node the branches meet at.
                let switch_name = spec.instance.clone().unwrap_or_else(|| {
                    alloc::format!("{FALLBACK_SWITCH_NAME}-{fallbacksrc_count}")
                });
                fallbacksrc_count += 1;
                let (main, fallback) = expand_fallbacksrc(
                    registry,
                    &spec,
                    consumer,
                    &switch_name,
                    avoided,
                    lone_fallbacksrc,
                )?;
                new_chain.extend(main);
                fallback_chains.push(fallback);
                continue;
            }
            let is_playbin = spec.name.starts_with("playbin");
            let uri =
                prop(&spec, "uri").ok_or_else(|| ParseError::MissingUri(spec.name.clone()))?;
            // `uridecodebin name=d  d. ! ...  d. ! ...` exposes every stream the URI
            // carries, as gst's does. A URI no fan-out hook claims has one stream.
            if !is_playbin && is_referenced(&spec, &referenced) {
                if let Some(fanout) = uri_fanout(registry, uri)? {
                    new_chain.push(Item::Prebuilt {
                        node: PrebuiltNode::UriFanout(fanout),
                        name: spec.instance.clone(),
                    });
                    continue;
                }
            }
            let built = registry
                .build_uri_source(uri)
                .map_err(|e: UriError| ParseError::Uri(alloc::format!("{uri}: {e:?}")))?;
            let caps = built.caps;
            // `playbin` names its own sink, a bare `uridecodebin` is followed by
            // one: either way the consumer's declared input memory picks the
            // decoder (M1018), as it does after a `decodebin`.
            let sink = is_playbin.then(|| {
                prop(&spec, "video-sink")
                    .unwrap_or(PlaybinSinks::AUTO.video)
                    .to_string()
            });
            let consumer = match (&sink, items.peek()) {
                (Some(sink), _) => Some(sink.as_str()),
                (None, Some((_, Item::Element(consumer)))) => Some(consumer.name.as_str()),
                _ => None,
            };
            let preferred = consumer.map_or(MemoryDomainKind::System, |name| {
                registry.declared_memory_preference(name)
            });
            let target = |c: &Caps| is_raw_video(c) || is_raw_audio(c);
            let decoders = registry
                .autoplug_avoiding(&caps, &target, DECODEBIN_MAX_DEPTH, preferred, avoided)
                .ok_or_else(|| ParseError::no_decode_chain(&caps))?;
            new_chain.push(Item::Prebuilt {
                node: PrebuiltNode::Source(built.source),
                name: None,
            });
            for dec in decoders {
                new_chain.push(Item::Prebuilt {
                    node: PrebuiltNode::Element(dec),
                    name: None,
                });
            }
            if built.has_clean_aperture {
                new_chain.push(clean_aperture_crop_item(registry)?);
            }
            // `d.` references read the decoded stream, so the name goes on the last node
            if !is_playbin {
                if let Some(Item::Prebuilt { name, .. }) = new_chain.last_mut() {
                    *name = spec.instance.clone();
                }
            }
            if let Some(sink) = sink {
                new_chain.push(Item::Element(ElementSpec {
                    name: sink,
                    props: Vec::new(),
                    instance: None,
                    log_category: None,
                }));
            }
        }
        out.push(new_chain);
        out.extend(fallback_chains);
    }
    Ok(out)
}

fn build_graph(
    registry: &Registry,
    chains: Vec<Chain>,
    avoided: &[&str],
) -> Result<Graph<GraphNode>, ParseError> {
    // Expand the source-providing (uridecodebin / playbin) and mid-chain
    // (decodebin) macros into concrete nodes before the structural build, so the
    // rest of the builder sees only real elements and pre-built nodes.
    let chains = expand_togglerecord(chains)?;
    let chains = expand_transcodebin(chains);
    let chains = expand_uri_sources(registry, chains, avoided)?;
    let chains = expand_decodebin(registry, chains, avoided)?;
    let chains = expand_encodebin(registry, chains)?;

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
    let mut prebuilt: Vec<Option<(PrebuiltNode, Option<String>)>> = Vec::new();
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
                Item::Prebuilt { node, name } => {
                    let ei = specs.len();
                    // A generated name collides with a line's own `name=` the
                    // same way the generated switch name does.
                    if let Some(inst) = &name {
                        if names.iter().any(|(n, _)| n == inst) {
                            return Err(ParseError::DuplicateName(inst.clone()));
                        }
                        names.push((inst.clone(), ei));
                    }
                    let placeholder = match node {
                        PrebuiltNode::Source(_) | PrebuiltNode::UriFanout(_) => "uridecodebin",
                        PrebuiltNode::Element(_) => "(decoder)",
                    };
                    specs.push(ElementSpec {
                        name: placeholder.to_string(),
                        props: Vec::new(),
                        instance: None,
                        log_category: None,
                    });
                    prebuilt.push(Some((node, name)));
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
    // The same requests as written, for a registered demux that routes by pad name.
    let mut output_pad_names: Vec<Vec<&str>> = alloc::vec![Vec::new(); specs.len()];
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
            output_pad_names[s].push(pads[w].unwrap_or(""));
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
    // The name feeding `ei` over its one inbound link, when it has exactly one.
    let sole_producer = |ei: usize| {
        let mut found = None;
        for &(s, d, _, _) in &links {
            if d != ei {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(specs[s].name.as_str());
        }
        found
    };
    // A non-tee node with several inbound links is a muxer (built from the
    // registry with that input count); a tee has a single input pad. A single
    // inbound link builds a fan-in too when the name is only registered as a
    // muxer (M1155: `livesync` needs the fan-in arm's deadline tick and has one
    // input). A name registered both ways (`mp4mux`, `textoverlay`) falls back
    // to its single-input element, unless that element does not take what the
    // one producer puts out and the fan-in does: a lone subtitle track into
    // `mp4mux` is the video-only `Mp4Mux` against `Mp4MuxN`'s `tx3g`.
    let is_muxer = |ei: usize| {
        !is_tee(ei)
            && match in_deg[ei] {
                0 => false,
                1 => {
                    registry.is_muxer(&specs[ei].name)
                        && (!registry.is_launch_element(&specs[ei].name)
                            || sole_producer(ei).is_some_and(|upstream| {
                                registry.launch_element_accepts(&specs[ei].name, upstream)
                                    == Some(false)
                            }))
                }
                _ => true,
            }
    };
    // A node registered as a demuxer with several outbound links is a fan-out
    // demux (M210): the transpose of a muxer. A registered name with one output
    // falls back to its single-output launch element (e.g. `tsdemux`), the way a
    // one-input muxer name registered both ways falls back to its single-input
    // element.
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
    // a demux behind a network source has no file to probe, so each port takes the
    // stream its consumer accepts, the way gst links a dynamic pad by caps
    for ei in 0..specs.len() {
        if demux_select_node[ei].is_some()
            || is_tee(ei)
            || is_demux(ei)
            || out_deg[ei] <= 1
            || prebuilt[ei].is_some()
        {
            continue;
        }
        let wanted: Option<Vec<CapsSet>> = links
            .iter()
            .filter(|(s, _, _, _)| *s == ei)
            .map(|(_, d, _, _)| registry.declared_input_caps(&specs[*d].name))
            .collect();
        let Some(wanted) = wanted else {
            continue;
        };
        demux_select_node[ei] = registry
            .demux_caps_select_hooks()
            .iter()
            .find_map(|hook| hook(&specs[ei].name, &wanted));
    }
    // `decodebin name=d` fan-out (M482): a `decodebin` node left unexpanded (it has
    // named refs) with a file source upstream probes the file, builds the
    // multi-output demuxer (stored like a demux-select node so it fans out on its
    // own pads), and records what each port carries so the wiring below splices
    // a decoder onto every port (the decode-per-port that makes it `decodebin`, not a
    // bare demuxer). Behind a network source the URI's fan-out supplies the demuxer
    // and its source replaces the line's.
    let mut decode_fanout_ports: Vec<Option<(AutoplugBin, Vec<UriFanoutPort>)>> =
        (0..specs.len()).map(|_| None).collect();
    // Which port each link out of a fan-out node reads, filled for every node
    // whose ports are matched to its branches.
    let mut fanout_links: Vec<Option<FanoutLinks>> = (0..specs.len()).map(|_| None).collect();
    let consumers_of = |ei: usize| -> Vec<Consumer<'_>> {
        links
            .iter()
            .filter(|(s, _, _, _)| *s == ei)
            .map(|(_, d, _, _)| Consumer {
                name: specs[*d].name.as_str(),
                is_muxer: is_muxer(*d),
            })
            .collect()
    };
    for ei in 0..specs.len() {
        let Some(bin) = autoplug_bin(&specs[ei].name) else {
            continue;
        };
        if out_deg[ei] == 0 || prebuilt[ei].is_some() {
            continue;
        }
        let upstream = links
            .iter()
            .find(|(_, d, _, _)| *d == ei)
            .map(|(s, _, _, _)| *s)
            .ok_or(ParseError::DecodebinNoUpstream)?;
        let upstream_spec = &specs[upstream];
        let location = prop(upstream_spec, "location");
        let unsupported = || {
            ParseError::DecodebinFanoutUpstream(match location {
                Some(location) => alloc::format!("{} location={location}", upstream_spec.name),
                None => upstream_spec.name.clone(),
            })
        };
        let location = location.ok_or_else(unsupported)?;
        let probed_file = registry
            .decodebin_select_hooks()
            .iter()
            .find_map(|hook| hook(location));
        let (demux, ports) = match probed_file {
            Some(found) => found,
            None if in_deg[upstream] == 0 => {
                let Some(UriFanout {
                    head:
                        UriFanoutHead::Demux {
                            mut source, demux, ..
                        },
                    ports,
                }) = uri_fanout(registry, location)?
                else {
                    return Err(unsupported());
                };
                apply_props(&mut source, &upstream_spec.name, &upstream_spec.props)?;
                prebuilt[upstream] =
                    Some((PrebuiltNode::Source(source), upstream_spec.instance.clone()));
                (demux, ports)
            }
            None => return Err(unsupported()),
        };
        let kinds: Vec<StreamType> = ports
            .iter()
            .map(|port| StreamType::of(&port.caps))
            .collect();
        fanout_links[ei] = Some(FanoutLinks {
            port_of_link: fanout_link_ports(
                registry,
                &specs[ei].name,
                &kinds,
                &demux_pads[ei],
                &output_pad_names[ei],
                &consumers_of(ei),
            )?,
            ports: kinds.len(),
        });
        demux_select_node[ei] = Some(demux);
        decode_fanout_ports[ei] = Some((bin, ports));
    }
    // `uridecodebin name=d` fan-out: the hook probed every port, so each `d.`
    // reference picks its port here and the wiring below decodes it.
    for ei in 0..specs.len() {
        let Some((PrebuiltNode::UriFanout(fanout), _)) = &prebuilt[ei] else {
            continue;
        };
        let kinds: Vec<StreamType> = fanout.ports.iter().map(|p| p.stream_type).collect();
        fanout_links[ei] = Some(FanoutLinks {
            port_of_link: fanout_link_ports(
                registry,
                &specs[ei].name,
                &kinds,
                &demux_pads[ei],
                &output_pad_names[ei],
                &consumers_of(ei),
            )?,
            ports: kinds.len(),
        });
        decode_fanout_ports[ei] = Some((AutoplugBin::Decode, fanout.ports.clone()));
    }
    let is_select: Vec<bool> = (0..specs.len())
        .map(|ei| demux_select_node[ei].is_some() || fanout_links[ei].is_some())
        .collect();
    // the single-stream form of a name registered both ways (`rtspsrc`) serves one
    // linked pad whose consumer takes the kind that form makes
    let single_source_serves = |ei: usize| {
        let made = registry
            .declared_output_caps(&specs[ei].name)
            .map(|caps| StreamType::of(&caps));
        out_deg[ei] <= 1
            && registry.is_source(&specs[ei].name)
            && consumers_of(ei)
                .into_iter()
                .all(|consumer| made.is_none_or(|kind| consumer_takes(registry, consumer, kind)))
    };
    let is_fanout_src = |ei: usize| {
        in_deg[ei] == 0 && registry.is_fanout_src(&specs[ei].name) && !single_source_serves(ei)
    };
    // Auto-tee (M473): a non-tee, non-demux node whose single output fans out to
    // several consumers gets an implicit `tee` spliced in below, so a gst-launch
    // line that omits the explicit tee still builds. `tee`, registered demuxers,
    // and explicit-demux fan-out nodes fan out on their own pads and are left alone.
    let needs_tee = |ei: usize| {
        !is_tee(ei) && !is_demux(ei) && !is_select[ei] && !is_fanout_src(ei) && out_deg[ei] > 1
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
        if let Some((node, name)) = prebuilt[ei].take() {
            let nid = match node {
                PrebuiltNode::Source(src) => graph.add_source(GraphNodeRef::Source(src)),
                PrebuiltNode::Element(el) if out_deg[ei] == 0 => {
                    graph.add_sink(GraphNodeRef::Element(el))
                }
                PrebuiltNode::Element(el) => graph.add_transform(GraphNodeRef::Element(el)),
                PrebuiltNode::UriFanout(fanout) => add_uri_fanout_head(&mut graph, fanout)?,
            };
            if let Some(name) = name {
                graph.set_node_name(nid, name);
            }
            node_of.push(Some(nid));
            continue;
        }
        let spec = &specs[ei];
        let node = if is_tee(ei) {
            graph.add_tee(out_deg[ei] as u8).node()
        } else if is_fanout_src(ei) {
            // Terminal fan-out source (M727): 0 inputs, one output per pad the
            // element has, each pad reference (`s. ! ...`) matched to one of
            // them. Properties are applied first: one of them may pick the pad
            // set (`rtspsrcn onvif-metadata=true` swaps the audio pad for the
            // metadata one).
            let mut src = registry
                .make_fanout_src(&spec.name, out_deg[ei])
                .ok_or_else(|| ParseError::UnknownElement(spec.name.clone()))?;
            apply_props(&mut src, &spec.name, &spec.props)?;
            // a pad whose caps are not known before the run takes any branch
            let kinds: Vec<StreamType> = (0..src.output_count())
                .map(|port| {
                    src.output_caps(port)
                        .map_or(StreamType::Unknown, |caps| StreamType::of(&caps))
                })
                .collect();
            fanout_links[ei] = Some(FanoutLinks {
                port_of_link: fanout_link_ports(
                    registry,
                    &spec.name,
                    &kinds,
                    &demux_pads[ei],
                    &output_pad_names[ei],
                    &consumers_of(ei),
                )?,
                ports: kinds.len(),
            });
            graph
                .add_fanout_src(GraphNodeRef::FanoutSource(src), kinds.len() as u8)
                .node()
        } else if in_deg[ei] == 0 {
            let src = build_source(registry, &spec.name, &spec.props)?;
            graph.add_source(GraphNodeRef::Source(src))
        } else if is_muxer(ei) {
            let mut mux = registry
                .make_muxer(&spec.name, in_deg[ei])
                .ok_or_else(|| ParseError::NotAMuxer(spec.name.clone()))?;
            apply_props(&mut mux, &spec.name, &spec.props)?;
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
            let ports = fanout_links[ei]
                .as_ref()
                .map_or(out_deg[ei], |fanout| fanout.ports);
            graph
                .add_demux(GraphNodeRef::Demux(demux), ports as u8)
                .node()
        } else if is_demux(ei) {
            let mut demux = registry
                .make_demux(&spec.name, out_deg[ei])
                .ok_or_else(|| ParseError::UnknownElement(spec.name.clone()))?;
            demux
                .name_output_ports(&output_pad_names[ei])
                .map_err(|port| ParseError::UnknownOutputPad {
                    element: spec.name.clone(),
                    pad: output_pad_names[ei]
                        .get(port)
                        .copied()
                        .unwrap_or_default()
                        .to_string(),
                })?;
            apply_props(&mut demux, &spec.name, &spec.props)?;
            graph
                .add_demux(GraphNodeRef::Demux(demux), out_deg[ei] as u8)
                .node()
        } else if out_deg[ei] == 0 {
            let el = build_element(registry, &spec.name, &spec.props)?;
            graph.add_sink(GraphNodeRef::Element(el))
        } else {
            let el = build_element(registry, &spec.name, &spec.props)?;
            graph.add_transform(GraphNodeRef::Element(el))
        };
        // M842: a `name=` is the element's instance name at run time, not just
        // the handle pad references resolve against.
        if let Some(inst) = &spec.instance {
            graph.set_node_name(node, inst.clone());
        }
        // M847: `log-category=` renames this instance's log category, which the
        // runner hands to the element before naming it.
        if let Some(cat) = &spec.log_category {
            graph.set_node_log_category(node, cat.clone());
        }
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
        } else if is_tee(s) || is_demux(s) || is_select[s] || is_fanout_src(s) {
            let ordinal = tee_next[s];
            tee_next[s] += 1;
            let index = fanout_links[s]
                .as_ref()
                .map_or(ordinal, |fanout| fanout.port_of_link[ordinal as usize]);
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
        if let Some((bin, ports)) = &decode_fanout_ports[s] {
            let port = &ports[src.index as usize];
            // the port's own caps say what it carries, a bare `d.` request does not
            let kind = StreamType::of(&port.caps);
            if kind != StreamType::Text {
                let target: &dyn Fn(&Caps) -> bool = match kind {
                    StreamType::Audio => &is_raw_audio,
                    _ => &is_raw_video,
                };
                match bin {
                    AutoplugBin::Decode => {
                        plug_port_decode(registry, &mut graph, src, dst, port, target)?
                    }
                    AutoplugBin::Parse => {
                        plug_port_parse(registry, &mut graph, src, dst, port, policy, capacity)?
                    }
                }
                continue;
            }
        }
        graph.link_full(src, dst, policy, capacity)?;
    }
    // gst leaves a stream no branch links unlinked and plays the rest, so it ends here
    for (ei, fanout) in fanout_links.iter().enumerate() {
        let (Some(fanout), Some(node)) = (fanout, node_of[ei]) else {
            continue;
        };
        for index in 0..fanout.ports as u8 {
            if fanout.port_of_link.contains(&index) {
                continue;
            }
            let sink = build_element(registry, UNLINKED_STREAM_SINK, &[])?;
            let sink = graph.add_sink(GraphNodeRef::Element(sink));
            graph.link(PadId { node, index }, sink)?;
        }
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
    parse_launch_avoiding(registry, pipeline, &[])
}

/// [`parse_launch`], with the auto-plug search forbidden from picking any
/// factory named in `avoided` (M1023). The line is otherwise parsed identically,
/// so an element the text names explicitly is still built: this bounds only what
/// a `decodebin` / `uridecodebin` / `playbin` may choose. An application retries
/// through here after a chosen decoder turned out not to decode the stream.
pub fn parse_launch_avoiding(
    registry: &Registry,
    pipeline: &str,
    avoided: &[&str],
) -> Result<Graph<GraphNode>, ParseError> {
    // Every element built below, including the ones a playbin hook builds, sees
    // this one id, and no other parse sees it.
    let _scope = ParseScope::enter();
    // The parser's own built-ins are not registry factories, but they are
    // element names all the same, so a chain may start on one.
    let knows = |name: &str| {
        registry.knows_element(name)
            || matches!(
                name,
                "queue"
                    | "queue2"
                    | "decodebin"
                    | "parsebin"
                    | "uridecodebin"
                    | "playbin"
                    | "encodebin"
                    | "encodebin2"
                    | "transcodebin"
            )
    };
    let chains = parse_chains_with(pipeline, &knows)?;
    // playbin uri=X auto-fan-out (M382): a lone `playbin uri=` probes the
    // container via the registered hook and auto-builds source -> demux ->
    // per-stream decode -> auto sinks (multi-stream). Without a hook (or if it
    // declines, e.g. a non-Matroska file), fall through to build_graph, which
    // expands `playbin` to the single-stream pipeline (M196).
    if let Some((uri, sinks)) = lone_playbin(&chains) {
        // Try each registered hook (one per container type) until one handles the
        // URI; a hook returns Ok(None) to decline a container it does not parse.
        for hook in registry.playbin_hooks() {
            if let Some(graph) = hook(registry, &sinks, uri)? {
                return Ok(splice_domain_converters(registry, graph));
            }
        }
    }
    // fallbacksrc uri=X per-kind fan-out (M1168): a lone `fallbacksrc` probes the
    // container the same way and gives every kind its own switch and sink. It
    // declines a single-kind container, which build_graph then expands to the
    // single-stream form plus one automatic sink.
    if let Some(spec) = lone_fallbacksrc_spec(&chains) {
        if let Some(graph) = build_fallbacksrc_fanout(registry, spec)? {
            return Ok(splice_domain_converters(registry, graph));
        }
    }
    Ok(splice_domain_converters(
        registry,
        build_graph(registry, chains, avoided)?,
    ))
}

/// The factory to bar from the auto-plug search after a run ended in
/// `failed_element`, or `None` when re-plugging cannot help (M1023). Pass the
/// result to [`parse_launch_avoiding`] (accumulating it into `avoided`) and run
/// the line again; `None` means report the failure instead.
///
/// The caller owns the rest of the policy, because only it knows whether a
/// restart is safe: retry only while nothing has been presented (a run that
/// output nothing restarts invisibly, one that did not would repeat itself) and
/// not for a live source ([`has_live_source`], where a restart reopens the
/// connection and drops what was in flight).
///
/// `None` when the failing element is unknown to the registry, when the line
/// names it explicitly or the application set its rank (the user asked for that
/// element, so substituting another would defy them), or when it has already
/// been barred once.
pub fn fallback_factory(
    registry: &Registry,
    pipeline: &str,
    failed_element: &str,
    avoided: &[&str],
) -> Option<&'static str> {
    let factory = registry.factory_of_instance(failed_element)?;
    if avoided.contains(&factory) {
        return None;
    }
    let named_in_line = tokenize(pipeline).iter().any(|t| t == factory);
    (!named_in_line && !registry.rank_overridden(factory)).then_some(factory)
}

/// Whether any element in `graph` offers a live-source clock: a capture or
/// network source, which a restart would reopen at the cost of whatever was in
/// flight. The signal a caller checks before re-running a pipeline it has already
/// started (see [`fallback_factory`]).
pub fn has_live_source(graph: &Graph<GraphNode>) -> bool {
    (0..graph.node_count())
        .filter_map(|i| graph.element(NodeId(i as u32)))
        .filter_map(|node| match node {
            GraphNodeRef::Source(source) => source.provide_clock(),
            GraphNodeRef::Element(element) => element.provide_clock(),
            _ => None,
        })
        .any(|candidate| candidate.priority == crate::clock::ClockPriority::LiveSource)
}

pub fn build_element(
    registry: &Registry,
    name: &str,
    properties: &[(String, String)],
) -> Result<Box<dyn DynAsyncElement>, ParseError> {
    let mut element = registry
        .make_element(name)
        .ok_or_else(|| ParseError::UnknownElement(name.to_string()))?;
    apply_props(&mut element, name, properties)?;
    Ok(element)
}

pub fn build_source(
    registry: &Registry,
    name: &str,
    properties: &[(String, String)],
) -> Result<Box<dyn DynSourceLoop>, ParseError> {
    let mut source = registry
        .make_source(name)
        .ok_or_else(|| ParseError::UnknownSource(name.to_string()))?;
    apply_props(&mut source, name, properties)?;
    Ok(source)
}

pub fn parse_element(
    registry: &Registry,
    description: &str,
) -> Result<Box<dyn DynAsyncElement>, ParseError> {
    let spec = parse_one_element(registry, description)?;
    build_element(registry, &spec.name, &spec.props)
}

pub fn parse_source(
    registry: &Registry,
    description: &str,
) -> Result<Box<dyn DynSourceLoop>, ParseError> {
    let spec = parse_one_element(registry, description)?;
    build_source(registry, &spec.name, &spec.props)
}

pub fn parse_chain(
    registry: &Registry,
    description: &str,
) -> Result<Vec<Box<dyn DynAsyncElement>>, ParseError> {
    parse_element_chain(registry, description)?
        .iter()
        .map(|spec| build_element(registry, &spec.name, &spec.props))
        .collect()
}

fn parse_one_element(registry: &Registry, description: &str) -> Result<ElementSpec, ParseError> {
    let mut specs = parse_element_chain(registry, description).map_err(|error| match error {
        ParseError::NotElementChain(description) => ParseError::NotOneElement(description),
        other => other,
    })?;
    if specs.len() != 1 {
        return Err(ParseError::NotOneElement(description.trim().to_string()));
    }
    Ok(specs.swap_remove(0))
}

fn parse_element_chain(
    registry: &Registry,
    description: &str,
) -> Result<Vec<ElementSpec>, ParseError> {
    let description = description.trim();
    let not_a_chain = || ParseError::NotElementChain(description.to_string());
    let tokens = tokenize(description);
    let mut tokens = tokens.iter().map(String::as_str).peekable();
    let mut specs = Vec::new();
    loop {
        let name = match tokens.next() {
            Some(name) => name,
            None if specs.is_empty() => return Err(ParseError::Empty),
            None => return Err(not_a_chain()),
        };
        if name == "!" || is_caps_token(name) || as_ref_name(name).is_some() {
            return Err(not_a_chain());
        }
        let spec = consume_element(name, &mut tokens, &|token| registry.knows_element(token))?;
        // the mutator names the instance itself
        let keyword = match (&spec.instance, &spec.log_category) {
            (Some(_), _) => Some("name"),
            (None, Some(_)) => Some("log-category"),
            (None, None) => None,
        };
        if let Some(keyword) = keyword {
            return Err(ParseError::LaunchKeyword {
                element: spec.name,
                key: keyword.to_string(),
            });
        }
        specs.push(spec);
        match tokens.next() {
            None => return Ok(specs),
            Some("!") => {}
            Some(_) => return Err(not_a_chain()),
        }
    }
}

pub async fn set_property_text(
    mutator: &GraphMutator<'_>,
    node: &str,
    name: &str,
    text: &str,
) -> Result<(), MutationError> {
    let value = match mutator.get_property(node, name).await? {
        Some(current) => {
            PropValue::parse(current.kind(), text).map_err(MutationError::PropertyRejected)?
        }
        // no value to take the kind from, so the element judges the text itself
        None => PropValue::Str(text.to_string()),
    };
    mutator.set_property(node, name, value).await
}

/// Splice memory-domain converters into a just-parsed graph (M1017), where the
/// registry carries a factory: a text pipeline that links a GPU decoder to a sink
/// in another domain gets the bridge (or the download) it needs without naming
/// it. A no-op when the domains already agree, or when no factory is registered.
fn splice_domain_converters(registry: &Registry, graph: Graph<GraphNode>) -> Graph<GraphNode> {
    match registry.domain_converter() {
        Some(factory) => crate::runtime::auto_plug_domain_converters(graph, &factory),
        None => graph,
    }
}

/// The `uri=` and sinks of a pipeline that is a single bare `playbin uri=X`
/// element (and nothing else), the M382 multi-stream auto-fan-out trigger.
/// `None` for any other shape: a `playbin` mid-pipeline, alongside other
/// elements, or without a `uri=` is left to the normal builder (the M196
/// single-stream expansion).
fn lone_playbin(chains: &[Chain]) -> Option<(&str, PlaybinSinks<'_>)> {
    let [chain] = chains else { return None };
    let [Item::Element(spec)] = chain.as_slice() else {
        return None;
    };
    if spec.name != "playbin" {
        return None;
    }
    let sinks = PlaybinSinks {
        video: prop(spec, "video-sink").unwrap_or(PlaybinSinks::AUTO.video),
        audio: prop(spec, "audio-sink").unwrap_or(PlaybinSinks::AUTO.audio),
    };
    Some((prop(spec, "uri")?, sinks))
}

/// The spec of a pipeline that is a single bare `fallbacksrc` (and nothing else),
/// the M1168 per-kind fan-out trigger. `None` for any other shape: an
/// inline `fallbacksrc` feeds user-written elements, and a launch line has no way
/// to name a second output, so it stays single-stream.
fn lone_fallbacksrc_spec(chains: &[Chain]) -> Option<&ElementSpec> {
    let [chain] = chains else { return None };
    let [Item::Element(spec)] = chain.as_slice() else {
        return None;
    };
    (spec.name == "fallbacksrc").then_some(spec)
}

/// Set a generated node name, rejecting a collision with one already generated
/// the way the parser rejects two elements sharing a `name=`.
fn name_node(
    graph: &mut Graph<GraphNode>,
    taken: &mut Vec<String>,
    node: NodeId,
    name: String,
) -> Result<(), ParseError> {
    if taken.contains(&name) {
        return Err(ParseError::DuplicateName(name));
    }
    graph.set_node_name(node, name.clone());
    taken.push(name);
    Ok(())
}

const UNLINKED_STREAM_SINK: &str = "fakesink";

struct FanoutLinks {
    port_of_link: Vec<u8>,
    ports: usize,
}

#[derive(Clone, Copy)]
struct Consumer<'a> {
    name: &'a str,
    is_muxer: bool,
}

// an element with no declared sink caps takes any stream, and a stream of unknown kind goes to any element
fn consumer_takes(registry: &Registry, consumer: Consumer<'_>, kind: StreamType) -> bool {
    if kind == StreamType::Unknown {
        return true;
    }
    // a name registered both ways declares its single-input element's sink caps, not the muxer's
    if consumer.is_muxer {
        return registry
            .muxer_input_kinds(consumer.name)
            .is_none_or(|kinds| kinds.contains(&kind));
    }
    registry
        .declared_input_caps(consumer.name)
        .is_none_or(|set| {
            set.alternatives()
                .iter()
                .any(|caps| StreamType::of(caps) == kind)
        })
}

// `d.video_1` reads the second video port and `d.src_1` the second port. A bare `d.` takes
// the first free port its consumer takes, as gst-launch's delayed link picks a pad by caps.
fn fanout_link_ports(
    registry: &Registry,
    element: &str,
    kinds: &[StreamType],
    requests: &[PadRequest],
    pad_names: &[&str],
    consumers: &[Consumer<'_>],
) -> Result<Vec<u8>, ParseError> {
    let mut taken = alloc::vec![false; kinds.len()];
    let mut named: Vec<Option<usize>> = Vec::with_capacity(requests.len());
    for (request, pad) in requests.iter().zip(pad_names) {
        if pad.is_empty() {
            named.push(None);
            continue;
        }
        let kind = match request.kind {
            PadKind::Video => Some(StreamType::Video),
            PadKind::Audio => Some(StreamType::Audio),
            PadKind::Text => Some(StreamType::Text),
            PadKind::Any => None,
        };
        let port = kinds
            .iter()
            .enumerate()
            .filter(|(_, port_kind)| kind.is_none_or(|kind| **port_kind == kind))
            .nth(request.index)
            .map(|(index, _)| index)
            .filter(|&index| !core::mem::replace(&mut taken[index], true))
            .ok_or_else(|| ParseError::UnknownOutputPad {
                element: element.to_string(),
                pad: pad.to_string(),
            })?;
        named.push(Some(port));
    }
    named
        .into_iter()
        .zip(consumers.iter().copied())
        .map(|(port, consumer)| {
            let port = match port {
                Some(port) => port,
                None => {
                    let port = (0..kinds.len())
                        .find(|&port| {
                            !taken[port] && consumer_takes(registry, consumer, kinds[port])
                        })
                        .ok_or_else(|| ParseError::NoFanoutStream {
                            element: element.to_string(),
                            consumer: consumer.name.to_string(),
                        })?;
                    taken[port] = true;
                    port
                }
            };
            Ok(port as u8)
        })
        .collect()
}

fn add_uri_fanout_head(
    graph: &mut Graph<GraphNode>,
    fanout: UriFanout,
) -> Result<NodeId, ParseError> {
    let ports = fanout.ports.len() as u8;
    Ok(match fanout.head {
        UriFanoutHead::Demux { source, demux, .. } => {
            let source = graph.add_source(GraphNodeRef::Source(source));
            let demux = graph.add_demux(GraphNodeRef::Demux(demux), ports);
            graph.link(source, demux.input())?;
            demux.node()
        }
        UriFanoutHead::FanoutSource { source, .. } => graph
            .add_fanout_src(GraphNodeRef::FanoutSource(source), ports)
            .node(),
    })
}

/// The first registered fan-out hook that claims `uri`, or `None` when every hook
/// declines (an unprobed scheme, or a container none of them parses).
fn uri_fanout(registry: &Registry, uri: &str) -> Result<Option<UriFanout>, ParseError> {
    for hook in registry.uri_fanout_hooks() {
        if let Some(fanout) = hook(registry, uri)? {
            return Ok(Some(fanout));
        }
    }
    Ok(None)
}

/// The branch shape per port of a fan-out, or `None` when any port's kind is one
/// a `fallbacksrc` has no generator and no sink for (text, unclassified), which
/// leaves the caller the single-stream expansion.
///
/// A kind may repeat (M1171): a container carrying two audio tracks gets two
/// audio branches. `ordinal` is how many earlier ports carried the same kind, so
/// each switch takes a distinct name and each main port pairs with the matching
/// fallback port rather than all of them with the first of their kind.
fn fallback_kinds(fanout: &UriFanout) -> Option<Vec<FanoutBranch>> {
    let mut seen: Vec<StreamType> = Vec::with_capacity(fanout.ports.len());
    let mut branches = Vec::with_capacity(fanout.ports.len());
    for port in &fanout.ports {
        let ordinal = seen.iter().filter(|t| **t == port.stream_type).count();
        seen.push(port.stream_type);
        branches.push(FanoutBranch {
            kind: fallback_kind(port.stream_type)?,
            ordinal,
        });
    }
    Some(branches)
}

/// The index of the `ordinal`-th port of `stream_type` in `ports`, or `None`
/// when there is no such port (M1171).
fn nth_port_of_kind(
    ports: &[UriFanoutPort],
    stream_type: StreamType,
    ordinal: usize,
) -> Option<usize> {
    ports
        .iter()
        .enumerate()
        .filter(|(_, p)| p.stream_type == stream_type)
        .map(|(i, _)| i)
        .nth(ordinal)
}

/// Whether every port in `fallback` has a `main` port of the same kind to feed,
/// counted per kind so two fallback audio tracks need two main audio tracks.
fn covers_every_port(main: &[UriFanoutPort], fallback: &[UriFanoutPort]) -> bool {
    let count = |ports: &[UriFanoutPort], t: StreamType| {
        ports.iter().filter(|p| p.stream_type == t).count()
    };
    fallback
        .iter()
        .all(|p| count(fallback, p.stream_type) <= count(main, p.stream_type))
}

enum FanoutFallback {
    PerPort {
        pads: Vec<PadId>,
        ports: Vec<UriFanoutPort>,
    },
    // `port` is the one main port an application-built source backs
    OnePort {
        pad: PadId,
        port: usize,
        stream: UriFanoutPort,
    },
}

// video before audio, as in the single-stream expansion
fn single_fallback_port(
    registry: &Registry,
    ports: &[UriFanoutPort],
    caps: &Caps,
) -> Option<usize> {
    [StreamType::Video, StreamType::Audio]
        .into_iter()
        .find_map(|stream_type| {
            let kind = fallback_kind(stream_type)?;
            let index = nth_port_of_kind(ports, stream_type, 0)?;
            registry
                .autoplug_names(caps, &kind.raw, DECODEBIN_MAX_DEPTH)
                .map(|_| index)
        })
}

/// One port's branch of a fanned-out `fallbacksrc`: its kind's shape, and which
/// port of that kind it is (M1171).
#[derive(Clone, Copy)]
struct FanoutBranch {
    kind: FallbackKind,
    ordinal: usize,
}

impl FanoutBranch {
    /// The name of this branch's switch: the keyword's stem plus the kind's
    /// suffix, and for the second and later port of a kind the ordinal too, so a
    /// two-audio-track container's switches never collide. The first port of a
    /// kind keeps the bare suffix, so a single-track container's names are the
    /// ones M1168 already generated.
    fn switch_name(&self, base: &str) -> String {
        match self.ordinal {
            0 => alloc::format!("{base}{}", self.kind.suffix),
            n => alloc::format!("{base}{}-{n}", self.kind.suffix),
        }
    }
}

/// Add one fan-out's head to `graph`, restart-wrapped and named `name`, and
/// answer the pad each port is read off (M1170). A file container's head is a
/// byte source into a demuxer, so the name goes on the source and the pads come
/// off the demuxer; a session's head is one multi-output source, so both are the
/// same node.
#[allow(clippy::too_many_arguments)]
fn add_fanout_head(
    registry: &Registry,
    graph: &mut Graph<GraphNode>,
    names: &mut Vec<String>,
    head: UriFanoutHead,
    ports: usize,
    settings: &FallbackSettings<'_>,
    role: FallbackSourceRole,
    name: String,
) -> Result<Vec<PadId>, ParseError> {
    match head {
        UriFanoutHead::Demux {
            source,
            rebuild,
            demux,
        } => {
            let source = restartable_source(
                registry,
                source,
                rebuild,
                settings.restart,
                role,
                settings.unblock.clone(),
            );
            let src = graph.add_source(GraphNodeRef::Source(source));
            name_node(graph, names, src, name)?;
            let demux = graph.add_demux(GraphNodeRef::Demux(demux), ports as u8);
            graph.link(src, demux.input())?;
            Ok((0..ports).map(|i| demux.out(i as u8)).collect())
        }
        UriFanoutHead::FanoutSource { source, rebuild } => {
            let source = restartable_fanout_source(
                registry,
                source,
                rebuild,
                settings.restart,
                role,
                settings.unblock.clone(),
            );
            let src = graph.add_fanout_src(GraphNodeRef::FanoutSource(source), ports as u8);
            name_node(graph, names, src.node(), name)?;
            Ok((0..ports).map(|i| src.output(i as u8)).collect())
        }
    }
}

/// Build the per-kind graph of a lone `fallbacksrc uri=X` (M1168): one restartable
/// byte source into one demuxer, then per demux port a decode chain into input 0
/// of that kind's `fallbackswitch`, the fallback branch into input 1, and the
/// switch output to that kind's automatic sink. An application-built fallback
/// (M1203) is one stream, so it backs one switch and the rest take dummies.
///
/// `Ok(None)` declines, leaving the single-stream expansion to build the line: no
/// hook claims the URI, the container carries one kind, a kind repeats, or either
/// `enable-video` / `enable-audio` is off.
fn build_fallbacksrc_fanout(
    registry: &Registry,
    spec: &ElementSpec,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let settings = fallbacksrc_settings(registry, spec)?;
    if !settings.enable_video || !settings.enable_audio {
        return Ok(None);
    }
    // An application-built source is one output with no container to probe.
    let FallbacksrcSourceOrigin::Uri(uri) = &settings.main else {
        return Ok(None);
    };
    let Some(main) = uri_fanout(registry, uri)? else {
        return Ok(None);
    };
    let Some(branches) = fallback_kinds(&main) else {
        return Ok(None);
    };
    if branches.len() < 2 {
        return Ok(None);
    }
    // A `fallbacksrc name=f` names its switches, so `f-video.` resolves to the
    // node the video branches meet at; unnamed it takes the generated stem the
    // single-stream expansion uses for its one switch.
    let base = spec
        .instance
        .clone()
        .unwrap_or_else(|| alloc::format!("{FALLBACK_SWITCH_NAME}-0"));
    // The fallback fan-out is used only when every port it reports has a main
    // port to feed; otherwise a port would have nothing to drain it, and the
    // unmatched branches take their dummy generators instead. Counted per kind,
    // so a two-audio-track fallback behind a one-audio-track main is refused.
    let fallback_fanout = match &settings.fallback {
        Some(FallbacksrcSourceOrigin::Uri(uri)) => uri_fanout(registry, uri)?
            .filter(|f| fallback_kinds(f).is_some() && covers_every_port(&main.ports, &f.ports)),
        Some(FallbacksrcSourceOrigin::Factory(_)) | None => None,
    };

    let mut graph: Graph<GraphNode> = Graph::new();
    let mut names: Vec<String> = Vec::new();
    let ports = main.ports;
    let main_pads = add_fanout_head(
        registry,
        &mut graph,
        &mut names,
        main.head,
        ports.len(),
        &settings,
        FallbackSourceRole::Main,
        alloc::format!("{base}{FALLBACK_MAIN_SOURCE_SUFFIX}"),
    )?;

    let fallback_name = alloc::format!("{base}{FALLBACK_FALLBACK_SOURCE_SUFFIX}");
    let fallback = match (fallback_fanout, &settings.fallback) {
        (Some(fanout), _) => {
            let pads = add_fanout_head(
                registry,
                &mut graph,
                &mut names,
                fanout.head,
                fanout.ports.len(),
                &settings,
                FallbackSourceRole::Fallback,
                fallback_name,
            )?;
            Some(FanoutFallback::PerPort {
                pads,
                ports: fanout.ports,
            })
        }
        (None, Some(origin @ FallbacksrcSourceOrigin::Factory(_))) => {
            let built = origin.build(
                registry,
                settings.restart,
                FallbackSourceRole::Fallback,
                settings.unblock.clone(),
            )?;
            let port = single_fallback_port(registry, &ports, &built.caps)
                .ok_or_else(|| ParseError::no_decode_chain(&built.caps))?;
            let src = graph.add_source(GraphNodeRef::Source(built.source));
            name_node(&mut graph, &mut names, src, fallback_name)?;
            Some(FanoutFallback::OnePort {
                pad: src.into(),
                port,
                stream: UriFanoutPort {
                    stream_type: StreamType::of(&built.caps),
                    caps: built.caps,
                    has_clean_aperture: built.has_clean_aperture,
                },
            })
        }
        (None, _) => None,
    };

    for (i, (port, branch)) in ports.iter().zip(&branches).enumerate() {
        let kind = &branch.kind;
        let mut switch = registry
            .make_muxer(FALLBACK_SWITCH, FALLBACK_SWITCH_INPUTS)
            .ok_or_else(|| ParseError::NotAMuxer(FALLBACK_SWITCH.to_string()))?;
        apply_props(&mut switch, FALLBACK_SWITCH, &settings.switch_props)?;
        let switch = graph.add_muxer(GraphNodeRef::Muxer(switch), FALLBACK_SWITCH_INPUTS as u8);
        name_node(
            &mut graph,
            &mut names,
            switch.node(),
            branch.switch_name(&base),
        )?;
        if let Some(category) = &spec.log_category {
            graph.set_node_log_category(switch.node(), category.clone());
        }
        plug_port_decode(
            registry,
            &mut graph,
            main_pads[i],
            switch.input(0),
            port,
            &kind.raw,
        )?;
        // The k-th main port of a kind pairs with the k-th fallback port of that
        // kind, so a two-track container's second audio stream backs the second
        // audio switch rather than every audio switch backing onto the first.
        let matching = match &fallback {
            Some(FanoutFallback::PerPort {
                pads,
                ports: fallback_ports,
            }) => nth_port_of_kind(fallback_ports, port.stream_type, branch.ordinal)
                .map(|j| (pads[j], &fallback_ports[j])),
            Some(FanoutFallback::OnePort {
                pad,
                port: backed,
                stream,
            }) => (*backed == i).then_some((*pad, stream)),
            None => None,
        };
        match matching {
            Some((pad, stream)) => plug_port_decode(
                registry,
                &mut graph,
                pad,
                switch.input(1),
                stream,
                &kind.raw,
            )?,
            None => plug_dummy(registry, &mut graph, switch.input(1), kind)?,
        }
        let mut head = {
            let sink = registry
                .make_element(kind.sink)
                .ok_or_else(|| ParseError::UnknownElement(kind.sink.to_string()))?;
            graph.add_sink(GraphNodeRef::Element(sink))
        };
        for name in kind.tail.iter().rev() {
            let element = registry
                .make_element(name)
                .ok_or_else(|| ParseError::UnknownElement((*name).to_string()))?;
            let node = graph.add_transform(GraphNodeRef::Element(element));
            graph.link(node, head)?;
            head = node;
        }
        graph.link(switch.output(), head)?;
    }
    Ok(Some(graph))
}

// the decode chain from the pad that carries `port` to `to`, or the parse fails naming its caps
fn plug_port_decode(
    registry: &Registry,
    graph: &mut Graph<GraphNode>,
    from: PadId,
    to: PadId,
    port: &UriFanoutPort,
    target: &dyn Fn(&Caps) -> bool,
) -> Result<(), ParseError> {
    registry
        .decodebin_port(graph, from, to, port, target, DECODEBIN_MAX_DEPTH)
        .map_err(|_| ParseError::no_decode_chain(&port.caps))?;
    Ok(())
}

// a bare `videocrop` for after the decoder of a stream whose pictures carry a crop rectangle
fn clean_aperture_crop_item(registry: &Registry) -> Result<Item, ParseError> {
    Ok(Item::Prebuilt {
        node: PrebuiltNode::Element(registry.clean_aperture_crop()?),
        name: None,
    })
}

/// Build the dummy fallback branch for one kind (black frames or silence behind
/// the pacer) into one switch input, the same generator the single-stream
/// expansion uses when there is no fallback source.
fn plug_dummy(
    registry: &Registry,
    graph: &mut Graph<GraphNode>,
    to: PadId,
    kind: &FallbackKind,
) -> Result<(), ParseError> {
    let (name, key, value) = kind.dummy;
    let mut source = registry
        .make_source(name)
        .ok_or_else(|| ParseError::UnknownSource(name.to_string()))?;
    apply_props(&mut source, name, &[(key.to_string(), value.to_string())])?;
    let source = graph.add_source(GraphNodeRef::Source(source));
    let pacer = registry
        .make_element(FALLBACK_PACER)
        .ok_or_else(|| ParseError::UnknownElement(FALLBACK_PACER.to_string()))?;
    let pacer = graph.add_transform(GraphNodeRef::Element(pacer));
    graph.link(source, pacer)?;
    graph.link(pacer, to)?;
    Ok(())
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
                Item::Prebuilt { .. } => "(prebuilt)",
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
    fn escaped_space_keeps_a_value_whole() {
        // gst-launch 1.26 does the same: `\ ` is a literal space, not a token
        // break (verified against `filesink location=`).
        assert_eq!(
            tokenize(r"filesink location=/my\ file.ts ! sink"),
            ["filesink", r"location=/my\ file.ts", "!", "sink"]
        );
        let chains = parse_chains(r"videotestsrc ! filesink location=/my\ file.ts").unwrap();
        let Item::Element(fs) = &chains[0][1] else {
            panic!("element")
        };
        assert_eq!(fs.props, [("location".to_string(), "/my file.ts".into())]);
    }

    #[test]
    fn escaped_quote_does_not_close_the_region() {
        // `"a\"b"` is one token holding `a"b`, as in gst-launch.
        assert_eq!(
            tokenize(r#"identity note="a\" b" ! sink"#),
            ["identity", r#"note="a\" b""#, "!", "sink"]
        );
        let chains = parse_chains(r#"videotestsrc ! identity note="a\" b""#).unwrap();
        let Item::Element(id) = &chains[0][1] else {
            panic!("element")
        };
        assert_eq!(id.props, [("note".to_string(), "a\" b".into())]);
    }

    #[test]
    fn quoted_region_may_sit_mid_value() {
        // gst-launch rejects this ("no element ..."): its lexer only quotes a
        // whole value. g2g accepts the superset, so a path with one spaced
        // component needs no quoting of the rest.
        let chains =
            parse_chains(r#"videotestsrc ! filesink location=/tmp/"my dir"/a.ts"#).unwrap();
        let Item::Element(fs) = &chains[0][1] else {
            panic!("element")
        };
        assert_eq!(
            fs.props,
            [("location".to_string(), "/tmp/my dir/a.ts".into())]
        );
    }

    #[test]
    fn unescaped_backslash_stays_literal() {
        // A Windows path survives unescaped. gst-launch would drop these
        // backslashes (`videos\a.ts` opens a file called `videosa.ts`), so this
        // is a deliberate deviation: `\` escapes only the launch-special
        // characters.
        assert_eq!(unquote_value(r"C:\videos\a.ts"), r"C:\videos\a.ts");
        assert_eq!(unquote_value(r"a\ b"), "a b");
        assert_eq!(unquote_value(r"a\\b"), r"a\b");
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
    fn enum_value_error_lists_the_valid_nicks() {
        let e = ParseError::BadEnumValue {
            element: "videoflip".to_string(),
            key: "method".to_string(),
            value: "sideways".to_string(),
            values: "none | clockwise",
        };
        let msg = alloc::format!("{e}");
        assert!(msg.contains("videoflip"), "{msg}");
        assert!(msg.contains("'method'"), "{msg}");
        assert!(msg.contains("valid: none | clockwise"), "{msg}");
        assert!(!msg.contains("flag set"), "no flag hint for a plain enum");
        // A malformed flag set (the whole value is reported) also hints at the
        // `+` syntax.
        let flags = ParseError::BadEnumValue {
            element: "rtspsrc".to_string(),
            key: "protocols".to_string(),
            value: "udp+".to_string(),
            values: "udp | tcp",
        };
        assert!(alloc::format!("{flags}").contains("flag set"));
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

    /// gst-launch runs several top-level chains in one pipeline
    /// (`videotestsrc ! xvimagesink audiotestsrc ! pulsesink`); a bare element
    /// name after a completed chain opens the next one. Before, the sink ate it
    /// as a property and reported it malformed.
    #[test]
    fn a_second_chain_may_follow_a_sink() {
        let known = |n: &str| matches!(n, "videotestsrc" | "fakesink" | "filesink");
        let chains = parse_chains_with("videotestsrc ! fakesink videotestsrc ! fakesink", &known)
            .expect("two chains parse");
        let shapes: Vec<Vec<&str>> = chains.iter().map(item_names).collect();
        assert_eq!(
            shapes,
            [["videotestsrc", "fakesink"], ["videotestsrc", "fakesink"]],
            "one chain per source"
        );
    }

    /// The property case must not regress: a bare token naming nothing
    /// registered is still the typo it looks like, not a new chain.
    #[test]
    fn a_bare_token_that_names_nothing_is_still_a_malformed_property() {
        let known = |n: &str| matches!(n, "videotestsrc" | "fakesink" | "filesink");
        assert!(matches!(
            parse_chains_with("videotestsrc bogus ! fakesink", &known),
            Err(ParseError::MalformedProperty { .. })
        ));
        // And after a sink, where the new-chain rule applies.
        assert!(matches!(
            parse_chains_with("videotestsrc ! fakesink bogus", &known),
            Err(ParseError::MalformedProperty { .. })
        ));
    }

    /// A property whose value happens to name an element is still a property:
    /// the `=` decides.
    #[test]
    fn a_key_value_token_is_never_a_chain_head() {
        let known = |n: &str| matches!(n, "videotestsrc" | "fakesink" | "filesink");
        let chains =
            parse_chains_with("videotestsrc ! filesink location=fakesink", &known).expect("parses");
        let shapes: Vec<Vec<&str>> = chains.iter().map(item_names).collect();
        assert_eq!(shapes, [["videotestsrc", "filesink"]], "one chain");
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

    /// M1155: an element whose output cadence is its own only gets the deadline
    /// tick on a fan-in arm, and it has one input, so a name registered only as a
    /// muxer builds a fan-in node at link degree one. A name registered both ways
    /// still builds its single-input element there.
    mod degree_one_fanin {
        use super::super::DynAsyncElement;
        use crate::memory::{MemoryDomain, SystemSlice};
        use crate::runtime::{
            block_on, parse_launch, run_graph, DynMultiInputElement, DynSourceLoop, LaunchFactory,
            MuxerFactory, Registry, SourceFactory, SourceLoop,
        };
        use crate::{
            AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError,
            MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
        };
        use alloc::boxed::Box;
        use core::future::Future;
        use core::pin::Pin;

        const FRAMES: u64 = 3;

        struct ZeroClock;
        impl PipelineClock for ZeroClock {
            fn now_ns(&self) -> u64 {
                0
            }
        }

        fn caps() -> Caps {
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(2),
                height: Dim::Fixed(2),
                framerate: Rate::Fixed(30 << 16),
                interlace: crate::Interlace::Any,
                colorimetry: crate::Colorimetry::UNKNOWN,
            }
        }

        struct CountSrc;

        impl SourceLoop for CountSrc {
            type RunFuture<'a>
                = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
            where
                Self: 'a;
            type CapsFuture<'a>
                = core::future::Ready<Result<Caps, G2gError>>
            where
                Self: 'a;

            fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
                core::future::ready(Ok(caps()))
            }

            fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
                Ok(ConfigureOutcome::Accepted)
            }

            fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
                Box::pin(async move {
                    for i in 0..FRAMES {
                        out.push(PipelinePacket::DataFrame(Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
                            FrameTiming::default(),
                            i,
                        )))
                        .await?;
                    }
                    out.push(PipelinePacket::Eos).await?;
                    Ok(FRAMES)
                })
            }
        }

        /// One input, one output, forwards what it gets: the shape an element
        /// needing the tick has.
        struct PassMux;

        impl MultiInputElement for PassMux {
            type ProcessFuture<'a>
                = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
            where
                Self: 'a;

            fn input_count(&self) -> usize {
                1
            }

            fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
                Ok(upstream.clone())
            }

            fn configure_pipeline(
                &mut self,
                _input: usize,
                _caps: &Caps,
            ) -> Result<ConfigureOutcome, G2gError> {
                Ok(ConfigureOutcome::Accepted)
            }

            fn output_caps(&self) -> Result<Caps, G2gError> {
                Ok(caps())
            }

            fn process<'a>(
                &'a mut self,
                _input: usize,
                packet: PipelinePacket,
                out: &'a mut dyn OutputSink,
            ) -> Self::ProcessFuture<'a> {
                Box::pin(async move {
                    if matches!(packet, PipelinePacket::DataFrame(_)) {
                        out.push(packet).await?;
                    }
                    Ok(())
                })
            }
        }

        /// The transform registration of the both-ways name. It drops every frame,
        /// so the sink count alone says which of the two registrations the parser
        /// picked.
        struct DropAll;

        impl AsyncElement for DropAll {
            type ProcessFuture<'a>
                = core::future::Ready<Result<(), G2gError>>
            where
                Self: 'a;

            fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
                Ok(upstream.clone())
            }

            fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
                Ok(ConfigureOutcome::Accepted)
            }

            fn process<'a>(
                &'a mut self,
                _packet: PipelinePacket,
                _out: &'a mut dyn OutputSink,
            ) -> Self::ProcessFuture<'a> {
                core::future::ready(Ok(()))
            }
        }

        struct Swallow;

        impl AsyncElement for Swallow {
            type ProcessFuture<'a>
                = core::future::Ready<Result<(), G2gError>>
            where
                Self: 'a;

            fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
                Ok(upstream.clone())
            }

            fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
                Ok(ConfigureOutcome::Accepted)
            }

            fn process<'a>(
                &'a mut self,
                _packet: PipelinePacket,
                _out: &'a mut dyn OutputSink,
            ) -> Self::ProcessFuture<'a> {
                core::future::ready(Ok(()))
            }
        }

        fn registry() -> Registry {
            let mut reg = Registry::new();
            reg.register_source(SourceFactory::new("countsrc", caps(), || {
                Box::new(CountSrc) as Box<dyn DynSourceLoop>
            }));
            reg.register_launch(LaunchFactory::new(
                "countsink",
                alloc::vec::Vec::new(),
                || Box::new(Swallow) as Box<dyn DynAsyncElement>,
            ));
            reg.register_muxer(MuxerFactory::new("passmux", |_inputs| {
                Box::new(PassMux) as Box<dyn DynMultiInputElement>
            }));
            // Registered both ways, like `mp4mux` / `textoverlay`.
            reg.register_muxer(MuxerFactory::new("bothways", |_inputs| {
                Box::new(PassMux) as Box<dyn DynMultiInputElement>
            }));
            reg.register_launch(LaunchFactory::new(
                "bothways",
                alloc::vec::Vec::new(),
                || Box::new(DropAll) as Box<dyn DynAsyncElement>,
            ));
            reg
        }

        #[test]
        fn a_muxer_only_name_builds_a_fan_in_on_one_inbound_link() {
            let reg = registry();
            let graph = parse_launch(&reg, "countsrc ! passmux ! countsink")
                .expect("a one-input muxer in a plain chain parses");
            let stats = block_on(run_graph(graph, &ZeroClock, 4)).expect("and runs");
            assert_eq!(
                stats.frames_consumed, FRAMES,
                "the fan-in forwarded every frame to the sink"
            );
        }

        #[test]
        fn a_name_registered_both_ways_stays_its_element_on_one_inbound_link() {
            let reg = registry();
            let graph = parse_launch(&reg, "countsrc ! bothways ! countsink").expect("parses");
            let stats = block_on(run_graph(graph, &ZeroClock, 4)).expect("runs");
            assert_eq!(
                stats.frames_consumed, 0,
                "the transform registration ran, and it drops every frame"
            );
        }
    }

    mod element_descriptions {
        use super::super::{parse_chain, parse_element, DynAsyncElement, ParseError};
        use crate::property::{PropError, PropKind, PropValue, PropertySpec};
        use crate::runtime::{LaunchFactory, Registry};
        use crate::{AsyncElement, Caps, ConfigureOutcome, G2gError, OutputSink, PipelinePacket};
        use alloc::boxed::Box;
        use alloc::string::ToString;

        const ELEMENT: &str = "leveler";
        const LEVEL: &str = "level";
        const LEVEL_SPECS: &[PropertySpec] = &[PropertySpec::new(LEVEL, PropKind::Int, "level")];

        #[derive(Default)]
        struct Leveler {
            level: i64,
        }

        impl AsyncElement for Leveler {
            type ProcessFuture<'a>
                = core::future::Ready<Result<(), G2gError>>
            where
                Self: 'a;

            fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
                Ok(upstream.clone())
            }

            fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
                Ok(ConfigureOutcome::Accepted)
            }

            fn properties(&self) -> &'static [PropertySpec] {
                LEVEL_SPECS
            }

            fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
                match name {
                    LEVEL => self.level = value.as_int().ok_or(PropError::Type)?,
                    _ => return Err(PropError::Unknown),
                }
                Ok(())
            }

            fn get_property(&self, name: &str) -> Option<PropValue> {
                (name == LEVEL).then_some(PropValue::Int(self.level))
            }

            fn process<'a>(
                &'a mut self,
                _packet: PipelinePacket,
                _out: &'a mut dyn OutputSink,
            ) -> Self::ProcessFuture<'a> {
                core::future::ready(Ok(()))
            }
        }

        fn registry() -> Registry {
            let mut reg = Registry::new();
            reg.register_launch(LaunchFactory::new(ELEMENT, alloc::vec::Vec::new(), || {
                Box::new(Leveler::default()) as Box<dyn DynAsyncElement>
            }));
            reg
        }

        #[test]
        fn builds_the_element_with_its_property() {
            let element = parse_element(&registry(), "leveler level=7").expect("parses");
            assert_eq!(element.get_property(LEVEL), Some(PropValue::Int(7)));
        }

        #[test]
        fn refuses_an_instance_name() {
            let error = parse_element(&registry(), "leveler name=l")
                .map(|_| ())
                .unwrap_err();
            assert_eq!(
                error,
                ParseError::LaunchKeyword {
                    element: ELEMENT.to_string(),
                    key: "name".to_string(),
                }
            );
        }

        #[test]
        fn refuses_a_link() {
            let error = parse_element(&registry(), "leveler ! leveler")
                .map(|_| ())
                .unwrap_err();
            assert!(matches!(error, ParseError::NotOneElement(_)), "{error:?}");
        }

        #[test]
        fn builds_a_chain_in_stream_order() {
            let chain =
                parse_chain(&registry(), "leveler level=1 ! leveler level=2").expect("parses");
            let levels: alloc::vec::Vec<_> = chain
                .iter()
                .map(|element| element.get_property(LEVEL))
                .collect();
            assert_eq!(levels, [Some(PropValue::Int(1)), Some(PropValue::Int(2))]);
        }

        #[test]
        fn refuses_a_caps_filter_in_a_chain() {
            let error = parse_chain(&registry(), "leveler ! video/x-raw ! leveler")
                .map(|_| ())
                .unwrap_err();
            assert!(matches!(error, ParseError::NotElementChain(_)), "{error:?}");
        }
    }
}
