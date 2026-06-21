//! Auto-plug: a runtime element registry plus a decode-chain search over the
//! static pad-template metadata (DESIGN.md §4.13.7, DESIGN_TODO "Auto-plug /
//! element registry / `decodebin`-equivalent"). M83.
//!
//! GStreamer's `decodebin` takes the caps coming off a source and walks the
//! registry for a chain of element factories whose pad templates compose from
//! that input down to raw, then instantiates the chain as a bin. We have the
//! type-level metadata already ([`PadTemplates`], [`PadTemplate`]) and a solver
//! that answers "can A's source feed B's sink?" ([`pad_link`]); what was
//! missing was (a) a runtime enumeration of element types and (b) the search
//! that composes their templates into an ordered chain.
//!
//! Two layers, split by what they need:
//! - **Search** (`runtime`, no_std + alloc). [`ElementDesc`] is a name + its
//!   pad templates; [`find_chain`] runs a breadth-first search over caps states,
//!   each edge an element whose sink accepts the current caps, until an
//!   element's source produces caps satisfying the target. Shortest chain wins.
//!   This is the intellectual core and is testable without constructing a
//!   single element.
//! - **Registry** (`std`). [`Registry`] pairs each [`ElementDesc`] with a
//!   parameterless factory producing a boxed [`DynAsyncElement`], so
//!   [`Registry::autoplug`] returns the instantiated chain ready to splice onto
//!   [`run_graph`](crate::runtime::run_graph) as a sub-graph of transforms.
//!
//! The search picks element *types*; it does not fixate geometry or framerate.
//! A decoder's source template is "raw video at any geometry", so the search
//! state stays open and the concrete values are chosen later at instance
//! negotiation when the chain is run. The target is therefore a shape predicate
//! (see [`is_raw_video`]), not a fixed caps.

use alloc::vec::Vec;

use crate::caps::{AudioFormat, Caps, CapsSet};
use crate::pad_template::{pad_link, PadCaps, PadDirection, PadTemplate};
use crate::runtime::solver::NegotiationFailure;

/// An element type's autoplug-relevant metadata: a display name and its static
/// pad templates (typically `<E as PadTemplates>::pad_templates()`). The search
/// reads only the first sink and first source template; multi-pad elements
/// (tees, muxers) are not auto-plug candidates and are simply never matched.
#[derive(Debug, Clone)]
pub struct ElementDesc {
    /// Human-readable type name, used to report the chosen chain.
    pub name: &'static str,
    /// The element type's pad templates, in declaration order.
    pub templates: Vec<PadTemplate>,
}

impl ElementDesc {
    /// Build a descriptor from a name and its pad templates.
    pub fn new(name: &'static str, templates: Vec<PadTemplate>) -> Self {
        Self { name, templates }
    }

    /// First sink (input) pad template, if any.
    fn sink(&self) -> Option<&PadTemplate> {
        self.templates.iter().find(|t| t.direction == PadDirection::Sink)
    }

    /// First source (output) pad template, if any.
    fn source(&self) -> Option<&PadTemplate> {
        self.templates.iter().find(|t| t.direction == PadDirection::Source)
    }

    /// If this element accepts caps shaped like `input` on its sink pad, the
    /// caps set its source pad can then produce; `None` if it has no sink or
    /// source pad, or its sink rejects `input`.
    ///
    /// Acceptance reuses the negotiation solver: `input` is wrapped as a
    /// producer and linked against the sink template. An `Unfixable` link (both
    /// sides still open, e.g. geometry `Any` feeding `Any`) counts as accepted,
    /// since the search resolves shapes, not concrete values, exactly as
    /// [`types_can_link`](crate::pad_template::types_can_link) does.
    fn step(&self, input: &Caps) -> Option<CapsSet> {
        let sink = self.sink()?;
        let source = self.source()?;
        let input_as_src = PadTemplate::source(CapsSet::one(input.clone()));
        match pad_link(&input_as_src, sink) {
            Ok(_) | Err(NegotiationFailure::Unfixable { .. }) => match &source.caps {
                PadCaps::Fixed(set) => Some(set.clone()),
                // A wildcard source pad produces nothing concrete to advance on.
                PadCaps::Any => None,
            },
            _ => None,
        }
    }
}

/// Shape predicate: the caps are raw (decoded) video. The canonical
/// `decodebin` target, "walk from this input down to raw video."
pub fn is_raw_video(caps: &Caps) -> bool {
    matches!(caps, Caps::RawVideo { .. })
}

/// Shape predicate: the caps are raw (decoded) PCM audio. The audio half of the
/// `decodebin` target, "walk down to raw audio." [`Caps::Audio`] is overloaded:
/// it also carries compressed AAC / Opus (the demuxer / parser output), so this
/// matches only the PCM formats, not a compressed stream still labelled `Audio`.
pub fn is_raw_audio(caps: &Caps) -> bool {
    matches!(caps, Caps::Audio { format: AudioFormat::PcmS16Le | AudioFormat::PcmF32Le, .. })
}

/// One element on an auto-plugged chain: which registered [`ElementDesc`] it is
/// (`index` into the searched slice) and the output caps the search chose for it
/// (the source-pad alternative it was matched to produce). The caps pin the
/// media type and format the element must emit, which a format-flexible element
/// (a converter, a multi-format decoder) needs to be constructed; geometry and
/// framerate may still be open and fixate later at instance negotiation.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLink {
    /// Index into the `descs` slice passed to [`find_chain`].
    pub index: usize,
    /// The caps this element was chosen to produce on its source pad.
    pub output: Caps,
}

/// Find the shortest chain of registered element types that converts `input`
/// caps into caps satisfying `target`, returning the chain in order (upstream
/// first): for each hop, the descriptor index and the output caps the search
/// picked for it.
///
/// Returns `Some(vec![])` if `input` already satisfies `target` (no elements
/// needed), or `None` if no chain exists within `max_depth` hops. The search is
/// breadth-first over caps states, so the first chain found is the shortest. An
/// element is never used twice on the same path, which terminates same-shape
/// loops (e.g. a parser whose sink and source are both H.264).
pub fn find_chain(
    descs: &[ElementDesc],
    input: &Caps,
    target: &dyn Fn(&Caps) -> bool,
    max_depth: usize,
) -> Option<Vec<ChainLink>> {
    if target(input) {
        return Some(Vec::new());
    }
    // BFS frontier: each entry is a reached caps state and the element path
    // that produced it. Depth is bounded by max_depth so an unsatisfiable
    // target terminates even with cycle-free same-shape elements.
    let mut frontier: Vec<(Caps, Vec<ChainLink>)> = Vec::from([(input.clone(), Vec::new())]);
    for _ in 0..max_depth {
        let mut next: Vec<(Caps, Vec<ChainLink>)> = Vec::new();
        for (caps, path) in &frontier {
            for (i, desc) in descs.iter().enumerate() {
                if path.iter().any(|link| link.index == i) {
                    continue;
                }
                let Some(out_set) = desc.step(caps) else { continue };
                for out in out_set.alternatives() {
                    let mut new_path = path.clone();
                    new_path.push(ChainLink { index: i, output: out.clone() });
                    if target(out) {
                        return Some(new_path);
                    }
                    next.push((out.clone(), new_path));
                }
            }
        }
        if next.is_empty() {
            return None;
        }
        frontier = next;
    }
    None
}

#[cfg(feature = "std")]
mod factory {
    use super::*;
    use alloc::boxed::Box;

    use alloc::string::String;

    use crate::element::{AsyncElement, DynAsyncElement};
    use crate::graph::{Graph, GraphError, NodeId, PadId};
    use crate::pad_template::{PadCaps, PadDirection, PadTemplate, PadTemplates};
    use crate::property::format_specs;
    use crate::runtime::{DynMultiInputElement, DynSourceLoop, GraphNode, GraphNodeRef};

    /// A registered element type: its autoplug metadata plus a constructor
    /// producing a boxed transform/sink for the graph runner. The constructor
    /// receives the output caps the search chose for this hop (see
    /// [`ChainLink::output`]), so a format-flexible element configures itself to
    /// produce the right format. It is a plain `fn` pointer, the common case
    /// being a non-capturing closure `|out| Box::new(MyTransform::new(out))`
    /// coerced at the call site; an element with a fixed output ignores the arg
    /// (`|_| Box::new(MyDecoder::new())`).
    pub struct ElementFactory {
        desc: ElementDesc,
        build: fn(&Caps) -> Box<dyn DynAsyncElement>,
    }

    impl ElementFactory {
        /// Register an element type by name, pad templates, and constructor.
        pub fn new(
            name: &'static str,
            templates: Vec<PadTemplate>,
            build: fn(&Caps) -> Box<dyn DynAsyncElement>,
        ) -> Self {
            Self { desc: ElementDesc::new(name, templates), build }
        }

        /// Build from a [`PadTemplates`] type, pulling its templates from the
        /// trait so the registration site names only the type and constructor.
        pub fn of<E: PadTemplates>(
            name: &'static str,
            build: fn(&Caps) -> Box<dyn DynAsyncElement>,
        ) -> Self {
            Self::new(name, E::pad_templates(), build)
        }

        /// Instantiate a fresh boxed element configured to produce `output`.
        pub fn build(&self, output: &Caps) -> Box<dyn DynAsyncElement> {
            (self.build)(output)
        }

        /// This factory's autoplug descriptor.
        pub fn desc(&self) -> &ElementDesc {
            &self.desc
        }
    }

    /// A named element factory for the `gst-launch` text parser and the
    /// `gst-inspect` dump (M105): a *parameterless* constructor plus the element's
    /// pad templates. Unlike [`ElementFactory`] (the autoplug factory, built from
    /// the chosen output caps), this default-constructs the element so the parser
    /// can then apply `key=value` properties to it, the
    /// `gst_element_factory_make` + `g_object_set` model.
    pub struct LaunchFactory {
        name: &'static str,
        templates: Vec<PadTemplate>,
        build: fn() -> Box<dyn DynAsyncElement>,
    }

    impl LaunchFactory {
        /// Register a transform / sink by name, pad templates, and a
        /// parameterless constructor (`|| Box::new(MyElement::new())`).
        pub fn new(
            name: &'static str,
            templates: Vec<PadTemplate>,
            build: fn() -> Box<dyn DynAsyncElement>,
        ) -> Self {
            Self { name, templates, build }
        }

        /// Build from a [`PadTemplates`] type, pulling its templates from the
        /// trait so the registration site names only the type and constructor.
        pub fn of<E: PadTemplates>(
            name: &'static str,
            build: fn() -> Box<dyn DynAsyncElement>,
        ) -> Self {
            Self::new(name, E::pad_templates(), build)
        }

        /// This factory's element name.
        pub fn name(&self) -> &'static str {
            self.name
        }
    }

    impl core::fmt::Debug for LaunchFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("LaunchFactory").field("name", &self.name).finish_non_exhaustive()
        }
    }

    /// A named fan-in muxer factory for the `gst-launch` parser (M122): an
    /// N-to-1 element built per use with the input count the parser derives from
    /// link degree. Unlike [`LaunchFactory`] (a single-in / single-out transform
    /// or sink), the constructor takes the input count, because a
    /// [`MultiInputElement`](crate::MultiInputElement)'s `input_count` must match
    /// the muxer node's input-pad count.
    pub struct MuxerFactory {
        name: &'static str,
        build: fn(usize) -> Box<dyn DynMultiInputElement>,
    }

    impl MuxerFactory {
        /// Register a fan-in muxer by name and an input-count constructor
        /// (`|n| Box::new(MyMux::new(n, ...))`).
        pub fn new(name: &'static str, build: fn(usize) -> Box<dyn DynMultiInputElement>) -> Self {
            Self { name, build }
        }

        /// This factory's element name.
        pub fn name(&self) -> &'static str {
            self.name
        }
    }

    impl core::fmt::Debug for MuxerFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("MuxerFactory").field("name", &self.name).finish_non_exhaustive()
        }
    }

    /// Format an element's pad templates the way `gst-inspect` lists them (one
    /// line per pad: direction + the caps it accepts/produces).
    fn format_templates(templates: &[PadTemplate]) -> String {
        use core::fmt::Write;
        let mut out = String::new();
        for t in templates {
            let dir = match t.direction {
                PadDirection::Sink => "SINK",
                PadDirection::Source => "SRC",
            };
            match &t.caps {
                PadCaps::Fixed(set) => {
                    let _ = writeln!(out, "  {dir}: {:?}", set.alternatives());
                }
                PadCaps::Any => {
                    let _ = writeln!(out, "  {dir}: ANY");
                }
            }
        }
        out
    }

    /// Why [`Registry::decodebin`] could not splice a chain.
    #[derive(Debug)]
    pub enum DecodebinError {
        /// No chain of registered elements converts the input caps to the target
        /// within the depth bound.
        NoChain,
        /// A graph link failed (e.g. a pad was out of range or already linked).
        Graph(GraphError),
    }

    impl From<GraphError> for DecodebinError {
        fn from(e: GraphError) -> Self {
            DecodebinError::Graph(e)
        }
    }

    /// The representative caps a `PadTemplates` type declares on its source pad:
    /// the first alternative of its first source template, or `None` if it has
    /// no source pad or only a wildcard one. This is what a g2g source "knows it
    /// produces" without byte-stream `typefind`, the input an auto-plugged
    /// decode chain starts from.
    pub fn declared_source_caps<S: PadTemplates>() -> Option<Caps> {
        match S::pad_template(PadDirection::Source)?.caps {
            PadCaps::Fixed(set) => set.alternatives().first().cloned(),
            PadCaps::Any => None,
        }
    }

    /// A registered source element: its declared output caps and a constructor.
    /// Unlike [`ElementFactory`] (transforms / sinks, which the search composes),
    /// a source is the *root* of a graph, so it carries its output caps directly
    /// rather than being matched into a chain. Use [`declared_source_caps`] to
    /// derive the caps from a [`PadTemplates`] type.
    pub struct SourceFactory {
        name: &'static str,
        output: Caps,
        build: fn() -> Box<dyn DynSourceLoop>,
    }

    impl SourceFactory {
        /// Register a source by name, its declared output caps, and constructor.
        pub fn new(name: &'static str, output: Caps, build: fn() -> Box<dyn DynSourceLoop>) -> Self {
            Self { name, output, build }
        }
    }

    impl core::fmt::Debug for SourceFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("SourceFactory")
                .field("name", &self.name)
                .field("output", &self.output)
                .finish_non_exhaustive()
        }
    }

    /// A parsed URI, split at `://` into a scheme and the remainder. The
    /// remainder is left uninterpreted: each [`UriSourceFactory`] reads it the
    /// way its scheme needs (a host:port for `udp://`, a filesystem path for
    /// `file://`, the whole URI for `rtsp://`). Minimal by design, so core pulls
    /// no URL-parsing dependency; scheme-specific parsing lives in the handler.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Uri<'a> {
        /// The full URI as given, e.g. `rtsp://host:554/stream`.
        pub raw: &'a str,
        /// The scheme before `://`, lowercased-by-convention by the caller.
        pub scheme: &'a str,
        /// Everything after `://`: authority + path + query, uninterpreted.
        pub rest: &'a str,
    }

    impl<'a> Uri<'a> {
        /// Split `raw` at the first `://`. `None` if there is no `://` or the
        /// scheme is empty.
        pub fn parse(raw: &'a str) -> Option<Uri<'a>> {
            let (scheme, rest) = raw.split_once("://")?;
            if scheme.is_empty() {
                return None;
            }
            Some(Uri { raw, scheme, rest })
        }
    }

    /// Why [`Registry::build_uridecodebin`] could not assemble a graph.
    #[derive(Debug)]
    pub enum UriError {
        /// The URI did not parse as `scheme://rest`, or a handler could not
        /// interpret its scheme-specific remainder (e.g. a bad `host:port`).
        Malformed,
        /// No URI handler is registered for the scheme.
        UnknownScheme,
        /// The source's caps could not be decoded to the target (wraps the
        /// `decodebin` failure).
        Decode(DecodebinError),
    }

    impl From<DecodebinError> for UriError {
        fn from(e: DecodebinError) -> Self {
            UriError::Decode(e)
        }
    }

    /// A URI handler's build function: parse a [`Uri`] into a constructed source
    /// plus the caps it produces (the `decodebin` input).
    type UriSourceBuild = fn(&Uri) -> Result<(Box<dyn DynSourceLoop>, Caps), UriError>;

    /// A URI-scheme handler: maps a parsed [`Uri`] to a constructed source and
    /// the source's declared output caps (the `decodebin` input). The analog of
    /// GStreamer's `GstURIHandler`. Unlike [`SourceFactory`] (a parameterless
    /// `playbin` root named directly), this builds the source *from the URI*, so
    /// `udp://host:port` and `file://path` configure themselves.
    pub struct UriSourceFactory {
        scheme: &'static str,
        build: UriSourceBuild,
    }

    impl UriSourceFactory {
        /// Register a handler for `scheme` (e.g. `"rtsp"`, `"udp"`, `"file"`).
        /// `build` parses the URI's remainder, constructs the source, and
        /// returns it with the caps it produces.
        pub fn new(scheme: &'static str, build: UriSourceBuild) -> Self {
            Self { scheme, build }
        }
    }

    impl core::fmt::Debug for UriSourceFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("UriSourceFactory").field("scheme", &self.scheme).finish_non_exhaustive()
        }
    }

    /// Why [`Registry::build_playbin`] could not assemble a graph.
    #[derive(Debug)]
    pub enum PlaybinError {
        /// No source is registered under the requested name.
        UnknownSource,
        /// The source's caps could not be decoded to the target (wraps the
        /// `decodebin` failure: no chain, or a graph link error).
        Decode(DecodebinError),
    }

    impl From<DecodebinError> for PlaybinError {
        fn from(e: DecodebinError) -> Self {
            PlaybinError::Decode(e)
        }
    }

    impl core::fmt::Debug for ElementFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("ElementFactory").field("name", &self.desc.name).finish_non_exhaustive()
        }
    }

    /// A runtime collection of element factories the auto-plugger searches over,
    /// the analog of GStreamer's plugin registry. Registration order is the
    /// tie-break only indirectly: [`find_chain`] is breadth-first, so among
    /// equal-length chains the one whose elements register earliest is found
    /// first.
    #[derive(Debug, Default)]
    pub struct Registry {
        factories: Vec<ElementFactory>,
        sources: Vec<SourceFactory>,
        uris: Vec<UriSourceFactory>,
        launch: Vec<LaunchFactory>,
        muxers: Vec<MuxerFactory>,
        /// gst-canonical-name aliases (M192): each maps a name to an ordered list
        /// of registered targets, the first that is actually registered wins. A
        /// plain rename is a one-entry list; `autovideosink` is a fallback chain
        /// (`waylandsink`, `kmssink`, ..., `fakesink`). Resolved at `make_*` time,
        /// so an alias whose targets are all feature-gated-out simply misses.
        aliases: Vec<(&'static str, &'static [&'static str])>,
    }

    impl Registry {
        /// An empty registry.
        pub fn new() -> Self {
            Self::default()
        }

        /// Register one element factory (a transform / sink the search composes
        /// into chains), returning `&mut self` to chain calls.
        pub fn register(&mut self, factory: ElementFactory) -> &mut Self {
            self.factories.push(factory);
            self
        }

        /// Register one source factory (a graph root for [`build_playbin`]),
        /// returning `&mut self` to chain calls.
        pub fn register_source(&mut self, source: SourceFactory) -> &mut Self {
            self.sources.push(source);
            self
        }

        /// Register one URI-scheme handler (a graph root for
        /// [`build_uridecodebin`](Self::build_uridecodebin)), returning
        /// `&mut self` to chain calls.
        pub fn register_uri(&mut self, handler: UriSourceFactory) -> &mut Self {
            self.uris.push(handler);
            self
        }

        /// Register a named transform / sink for the `gst-launch` parser and
        /// `gst-inspect` (M105), returning `&mut self` to chain calls.
        pub fn register_launch(&mut self, factory: LaunchFactory) -> &mut Self {
            self.launch.push(factory);
            self
        }

        /// Register a named fan-in muxer for the `gst-launch` parser (M122),
        /// returning `&mut self` to chain calls.
        pub fn register_muxer(&mut self, factory: MuxerFactory) -> &mut Self {
            self.muxers.push(factory);
            self
        }

        /// Register a gst-canonical-name alias (M192): `name` resolves, at
        /// `make_source` / `make_element` time, to the first of `targets` that is
        /// actually registered. Use a one-entry list for a plain rename
        /// (`avdec_h264` -> `ffmpegdec`) and a fallback chain for an auto element
        /// (`autovideosink` -> `["waylandsink", "kmssink", "fakesink"]`). Returns
        /// `&mut self` to chain calls.
        pub fn register_alias(
            &mut self,
            name: &'static str,
            targets: &'static [&'static str],
        ) -> &mut Self {
            self.aliases.push((name, targets));
            self
        }

        /// Resolve a name through the alias table to the first registered target,
        /// or the name itself when it is not an alias. One hop only (aliases do not
        /// chain to other aliases).
        fn resolve_alias<'a>(&self, name: &'a str) -> &'a str {
            if let Some((_, targets)) = self.aliases.iter().find(|(a, _)| *a == name) {
                for &t in *targets {
                    if self.sources.iter().any(|s| s.name == t)
                        || self.launch.iter().any(|f| f.name == t)
                    {
                        return t;
                    }
                }
            }
            name
        }

        /// Construct a registered source by name (the parser's first element).
        /// `None` if no source is registered under `name` (after alias resolution).
        pub fn make_source(&self, name: &str) -> Option<Box<dyn DynSourceLoop>> {
            let name = self.resolve_alias(name);
            self.sources.iter().find(|s| s.name == name).map(|s| (s.build)())
        }

        /// Construct a registered transform / sink by name (a parser interior or
        /// tail element), default-configured. `None` if `name` is not registered
        /// via [`register_launch`](Self::register_launch) (after alias resolution).
        pub fn make_element(&self, name: &str) -> Option<Box<dyn DynAsyncElement>> {
            let name = self.resolve_alias(name);
            self.launch.iter().find(|f| f.name == name).map(|f| (f.build)())
        }

        /// The caps a registered element is known to produce on its source pad,
        /// without constructing or negotiating it: a source's declared output, or
        /// a transform / sink's first fixed source-pad template alternative.
        /// `None` for an unregistered name or one whose source pad is wildcard.
        /// The `decodebin` parser uses this to learn its upstream caps (the input
        /// to the auto-plug search). Reads the factory-declared media type; it does
        /// not reflect instance properties that re-type the output (e.g. a
        /// `filesrc`'s `bytestream-format`).
        pub fn declared_output_caps(&self, name: &str) -> Option<Caps> {
            let name = self.resolve_alias(name);
            if let Some(s) = self.sources.iter().find(|s| s.name == name) {
                return Some(s.output.clone());
            }
            let f = self.launch.iter().find(|f| f.name == name)?;
            let t = f.templates.iter().find(|t| t.direction == PadDirection::Source)?;
            match &t.caps {
                PadCaps::Fixed(set) => set.alternatives().first().cloned(),
                PadCaps::Any => None,
            }
        }

        /// Construct a registered fan-in muxer by name with `inputs` input pads
        /// (the parser derives the count from link degree, so it matches the
        /// muxer node's input-pad count). `None` if `name` is not registered via
        /// [`register_muxer`](Self::register_muxer).
        pub fn make_muxer(&self, name: &str, inputs: usize) -> Option<Box<dyn DynMultiInputElement>> {
            self.muxers.iter().find(|m| m.name == name).map(|m| (m.build)(inputs))
        }

        /// The names of every element registerable by the parser: sources first,
        /// then transforms / sinks, each in registration order. The `gst-inspect`
        /// element list.
        pub fn element_names(&self) -> Vec<&'static str> {
            self.sources
                .iter()
                .map(|s| s.name)
                .chain(self.launch.iter().map(|f| f.name))
                .chain(self.muxers.iter().map(|m| m.name))
                .collect()
        }

        /// One line per registerable element, `name: Long-name` (the long name
        /// from the element's [`metadata`](crate::AsyncElement::metadata), or just
        /// the name when it declares none), for the `gst-inspect` element index.
        /// Sources, then transforms / sinks, then muxers. Each non-muxer element is
        /// default-built to read its metadata (side-effect-free, like
        /// [`inspect`](Self::inspect)).
        pub fn element_listing(&self) -> Vec<String> {
            use alloc::string::ToString;
            let line = |name: &str, long: &str| {
                if long.is_empty() {
                    name.to_string()
                } else {
                    let mut s = name.to_string();
                    s.push_str(": ");
                    s.push_str(long);
                    s
                }
            };
            let mut lines = Vec::new();
            for s in &self.sources {
                lines.push(line(s.name, (s.build)().metadata().long_name));
            }
            for f in &self.launch {
                lines.push(line(f.name, (f.build)().metadata().long_name));
            }
            for m in &self.muxers {
                lines.push(m.name.to_string());
            }
            lines
        }

        /// A `gst-inspect`-style dump for the named element: its role, its
        /// settable properties, and (for a transform / sink) its pad templates.
        /// `None` if the name is not registered. The element is default-built to
        /// read its property table (the specs are `&'static`, behind an instance
        /// method), so building must be side-effect-free, as the in-tree
        /// constructors are.
        pub fn inspect(&self, name: &str) -> Option<String> {
            use core::fmt::Write;
            use crate::property::format_metadata;
            let mut out = String::new();
            if let Some(s) = self.sources.iter().find(|s| s.name == name) {
                let src = (s.build)();
                out.push_str(&format_metadata(name, &src.metadata()));
                let _ = writeln!(out, "  Role        source");
                let _ = writeln!(out, "\nOutput caps:\n  {:?}", s.output);
                let _ = write!(out, "\nElement Properties:\n{}", format_specs(src.properties()));
                Some(out)
            } else if let Some(f) = self.launch.iter().find(|f| f.name == name) {
                let el = (f.build)();
                out.push_str(&format_metadata(name, &el.metadata()));
                let _ = writeln!(out, "  Role        element");
                let _ = write!(out, "\nPad Templates:\n{}", format_templates(&f.templates));
                let _ = write!(out, "\nElement Properties:\n{}", format_specs(el.properties()));
                Some(out)
            } else if let Some(m) = self.muxers.iter().find(|m| m.name == name) {
                let _ = writeln!(out, "Factory Details:");
                let _ = writeln!(out, "  Name        {}", m.name);
                let _ = writeln!(out, "  Role        muxer (fan-in)");
                let _ = writeln!(out, "\nInputs: derived from link degree");
                Some(out)
            } else {
                None
            }
        }

        /// The descriptors of every registered factory, in registration order,
        /// indexed identically to the [`find_chain`] result.
        fn descs(&self) -> Vec<ElementDesc> {
            self.factories.iter().map(|f| f.desc.clone()).collect()
        }

        /// The names of the shortest chain converting `input` into caps
        /// satisfying `target`, without instantiating anything. `Some(vec![])`
        /// if `input` already satisfies `target`; `None` if no chain exists
        /// within `max_depth`.
        pub fn autoplug_names(
            &self,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
        ) -> Option<Vec<&'static str>> {
            let descs = self.descs();
            let chain = find_chain(&descs, input, target, max_depth)?;
            Some(chain.into_iter().map(|link| self.factories[link.index].desc.name).collect())
        }

        /// Find the shortest chain converting `input` into caps satisfying
        /// `target` and instantiate it: an ordered list of boxed elements
        /// (upstream first), each configured to produce the caps the search
        /// chose for it, ready to splice onto [`run_graph`] as transforms.
        /// `Some(vec![])` if no elements are needed; `None` if no chain exists.
        ///
        /// [`run_graph`]: crate::runtime::run_graph
        pub fn autoplug(
            &self,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
        ) -> Option<Vec<Box<dyn DynAsyncElement>>> {
            let descs = self.descs();
            let chain = find_chain(&descs, input, target, max_depth)?;
            Some(
                chain
                    .into_iter()
                    .map(|link| self.factories[link.index].build(&link.output))
                    .collect(),
            )
        }

        /// `decodebin`-equivalent: auto-plug a decode chain and splice it into
        /// `graph` as a run of transforms between an existing output pad `from`
        /// (which produces `input` caps) and an existing input pad `to`. Returns
        /// the inserted transform node ids in chain order.
        ///
        /// This is the "returns a sub-graph onto `run_graph`" payoff: the caller
        /// builds its source and sink, names the input caps and the target shape
        /// ([`is_raw_video`] for playback), and the registry fills the middle. An
        /// empty chain (input already satisfies `target`) links `from` straight
        /// to `to`.
        pub fn decodebin(
            &self,
            graph: &mut Graph<GraphNode>,
            from: impl Into<PadId>,
            to: impl Into<PadId>,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
        ) -> Result<Vec<NodeId>, DecodebinError> {
            let elements = self.autoplug(input, target, max_depth).ok_or(DecodebinError::NoChain)?;
            let mut prev: PadId = from.into();
            let to: PadId = to.into();
            let mut inserted = Vec::with_capacity(elements.len());
            for boxed in elements {
                let node = graph.add_transform(GraphNodeRef::Element(boxed));
                graph.link(prev, node)?;
                inserted.push(node);
                prev = node.into();
            }
            graph.link(prev, to)?;
            Ok(inserted)
        }

        /// `playbin`-equivalent: assemble a complete runnable graph from a
        /// registered source name and a sink, auto-plugging the decode chain in
        /// between. Looks up the source factory, takes its declared output caps
        /// as the `decodebin` input, and returns `source -> chain -> sink` ready
        /// for [`run_graph`](crate::runtime::run_graph). This is the "just play
        /// this" entry point, minus the URI-scheme front door (the caller still
        /// names the source rather than passing a `uri=`).
        pub fn build_playbin<Sk: AsyncElement + 'static>(
            &self,
            source_name: &str,
            sink: Sk,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
        ) -> Result<Graph<GraphNode>, PlaybinError> {
            let source = self
                .sources
                .iter()
                .find(|s| s.name == source_name)
                .ok_or(PlaybinError::UnknownSource)?;
            let mut graph: Graph<GraphNode> = Graph::new();
            let src = graph.add_source(GraphNodeRef::Source((source.build)()));
            let snk = graph.add_sink(GraphNodeRef::element(sink));
            self.decodebin(&mut graph, src, snk, &source.output, target, max_depth)?;
            Ok(graph)
        }

        /// `uridecodebin`-equivalent: the URI-scheme front door to
        /// [`build_playbin`](Self::build_playbin). Parses `uri`, dispatches to
        /// the registered [`UriSourceFactory`] for its scheme to construct the
        /// source from the URI, then auto-plugs `source -> chain -> sink` down
        /// to `target`, returning a graph ready for
        /// [`run_graph`](crate::runtime::run_graph).
        ///
        /// `target` is a shape predicate (commonly [`is_raw_video`] for
        /// playback); the source's runtime caps are resolved at negotiation, so
        /// the handler's declared output caps only need to name the *media type*
        /// the right decoder is plugged for.
        pub fn build_uridecodebin<Sk: AsyncElement + 'static>(
            &self,
            uri: &str,
            sink: Sk,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
        ) -> Result<Graph<GraphNode>, UriError> {
            let parsed = Uri::parse(uri).ok_or(UriError::Malformed)?;
            let handler = self
                .uris
                .iter()
                .find(|h| h.scheme == parsed.scheme)
                .ok_or(UriError::UnknownScheme)?;
            let (source, output) = (handler.build)(&parsed)?;
            let mut graph: Graph<GraphNode> = Graph::new();
            let src = graph.add_source(GraphNodeRef::Source(source));
            let snk = graph.add_sink(GraphNodeRef::element(sink));
            self.decodebin(&mut graph, src, snk, &output, target, max_depth)?;
            Ok(graph)
        }
    }
}

#[cfg(feature = "std")]
pub use factory::{
    declared_source_caps, DecodebinError, ElementFactory, LaunchFactory, MuxerFactory,
    PlaybinError, Registry, SourceFactory, Uri, UriError, UriSourceFactory,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, RawVideoFormat, VideoCodec};

    fn h264(width: Dim) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn raw(format: RawVideoFormat) -> Caps {
        Caps::RawVideo { format, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
    }

    /// H.264 in, H.264 out (a parser: refines but never changes media type).
    fn parser() -> ElementDesc {
        ElementDesc::new(
            "h264parse",
            Vec::from([
                PadTemplate::sink(CapsSet::one(h264(Dim::Any))),
                PadTemplate::source(CapsSet::one(h264(Dim::Any))),
            ]),
        )
    }

    /// H.264 in, raw NV12 out (a decoder).
    fn decoder() -> ElementDesc {
        ElementDesc::new(
            "h264dec",
            Vec::from([
                PadTemplate::sink(CapsSet::one(h264(Dim::Any))),
                PadTemplate::source(CapsSet::one(raw(RawVideoFormat::Nv12))),
            ]),
        )
    }

    /// Raw NV12 in, raw RGBA out (a converter).
    fn convert() -> ElementDesc {
        ElementDesc::new(
            "videoconvert",
            Vec::from([
                PadTemplate::sink(CapsSet::one(raw(RawVideoFormat::Nv12))),
                PadTemplate::source(CapsSet::one(raw(RawVideoFormat::Rgba8))),
            ]),
        )
    }

    /// Just the descriptor indices of a found chain, for terse assertions.
    fn indices(chain: &[ChainLink]) -> Vec<usize> {
        chain.iter().map(|l| l.index).collect()
    }

    #[test]
    fn finds_single_decoder_for_h264_to_raw() {
        let descs = [parser(), decoder()];
        let chain = find_chain(&descs, &h264(Dim::Fixed(1280)), &is_raw_video, 4)
            .expect("decoder bridges H.264 to raw");
        // Shortest path is the decoder alone (the parser is same-shape, so it
        // never shortens the route to raw), and it was chosen to emit NV12.
        assert_eq!(indices(&chain), Vec::from([1usize]));
        assert_eq!(chain[0].output, raw(RawVideoFormat::Nv12));
    }

    #[test]
    fn empty_chain_when_input_already_satisfies_target() {
        let descs = [decoder()];
        let chain = find_chain(&descs, &raw(RawVideoFormat::Nv12), &is_raw_video, 4)
            .expect("already raw");
        assert!(chain.is_empty(), "no elements needed when input is already raw");
    }

    #[test]
    fn finds_multi_element_chain_to_a_specific_format() {
        // Target a format only the converter produces, forcing decoder -> convert.
        let descs = [parser(), decoder(), convert()];
        let target = |c: &Caps| matches!(c, Caps::RawVideo { format: RawVideoFormat::Rgba8, .. });
        let chain = find_chain(&descs, &h264(Dim::Any), &target, 4)
            .expect("decode then convert reaches RGBA");
        assert_eq!(indices(&chain), Vec::from([1usize, 2usize]), "decoder then converter");
        // The converter hop carries the chosen output the builder needs: RGBA.
        assert_eq!(chain.last().unwrap().output, raw(RawVideoFormat::Rgba8));
    }

    #[test]
    fn no_chain_when_target_unreachable() {
        // Only a parser is registered: H.264 can never become raw.
        let descs = [parser()];
        assert!(
            find_chain(&descs, &h264(Dim::Any), &is_raw_video, 8).is_none(),
            "a parser alone cannot reach raw video"
        );
    }

    #[test]
    fn respects_max_depth() {
        // The decoder -> convert chain is length 2; a depth bound of 1 can't
        // reach the RGBA-only target.
        let descs = [decoder(), convert()];
        let target = |c: &Caps| matches!(c, Caps::RawVideo { format: RawVideoFormat::Rgba8, .. });
        assert!(find_chain(&descs, &h264(Dim::Any), &target, 1).is_none(), "1 hop is too shallow");
        assert!(find_chain(&descs, &h264(Dim::Any), &target, 2).is_some(), "2 hops suffice");
    }

    #[cfg(feature = "std")]
    #[test]
    fn uri_parse_splits_scheme_and_rest() {
        let u = Uri::parse("rtsp://cam.local:554/stream1?tcp").expect("valid uri");
        assert_eq!(u.scheme, "rtsp");
        assert_eq!(u.rest, "cam.local:554/stream1?tcp");
        assert_eq!(u.raw, "rtsp://cam.local:554/stream1?tcp");

        let f = Uri::parse("file:///home/a/clip.mp4").expect("valid file uri");
        assert_eq!(f.scheme, "file");
        assert_eq!(f.rest, "/home/a/clip.mp4", "file:// leaves an absolute path");

        let udp = Uri::parse("udp://0.0.0.0:5004").expect("valid udp uri");
        assert_eq!((udp.scheme, udp.rest), ("udp", "0.0.0.0:5004"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn uri_parse_rejects_malformed() {
        assert!(Uri::parse("notauri").is_none(), "no scheme separator");
        assert!(Uri::parse("://nohost").is_none(), "empty scheme");
    }
}
