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
use crate::memory::MemoryDomainKind;
use crate::pad_template::{pad_link, PadCaps, PadDirection, PadTemplate};
use crate::runtime::solver::NegotiationFailure;

/// Whether an element's path is hardware-accelerated. This is independent of
/// where the output frames land ([`CapabilityDescriptor::output_memory`]): an
/// ffmpeg VA-API decoder is hardware yet downloads to `System`, while a CPU
/// decoder is software and `System`. Auto-plug can prefer / avoid hardware per
/// request (throughput vs power) separately from the memory domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Acceleration {
    /// CPU / pure-software path (the default).
    #[default]
    Software,
    /// Fixed-function / GPU / platform hardware decode or encode.
    Hardware,
}

/// What an element offers the auto-plug search beyond its pad templates: the
/// signals used to choose among several elements that satisfy the same caps.
///
/// This generalizes a bare output-memory tag, the principled alternative to a
/// flat global rank. A single integer cannot express that the best element is
/// context-dependent (a hardware decoder that keeps frames on the GPU beats a
/// faster one that forces a PCIe download when the consumer is GPU-resident), so
/// [`score`](Self::score) ranks a candidate against a [`SelectionContext`]; the
/// numeric [`rank`](Self::rank) is only the deterministic tiebreaker among
/// otherwise-equal candidates (the explicit-override knob GStreamer's rank gets
/// right). All-default descriptors score equally, so registration order decides,
/// leaving a plain pipeline's selection unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityDescriptor {
    /// The memory feature the source pad emits (a `decodebin` caps feature,
    /// GStreamer's `memory:CUDAMemory` analog). Caps geometry / format is in the
    /// pad templates; the *memory domain* is not, so it rides here. Matching the
    /// consumer's domain avoids a copy / download (e.g. pick `NvDec` -> `Cuda`).
    pub output_memory: MemoryDomainKind,
    /// Hardware vs software path.
    pub acceleration: Acceleration,
    /// Deterministic tiebreaker among candidates of equal score (higher wins);
    /// default 0. Use it to order otherwise-equivalent backends (e.g. two ffmpeg
    /// codecs), or as an operator override, not as the primary selector.
    pub rank: i16,
}

impl Default for CapabilityDescriptor {
    fn default() -> Self {
        Self {
            output_memory: MemoryDomainKind::System,
            acceleration: Acceleration::Software,
            rank: 0,
        }
    }
}

impl CapabilityDescriptor {
    /// Score this candidate against what the search wants; higher is tried first.
    /// Memory-domain match dominates (it removes a download, the structural win),
    /// then a hardware preference, then the explicit [`rank`](Self::rank).
    pub fn score(&self, ctx: &SelectionContext) -> i32 {
        let mut s = 0i32;
        if self.output_memory == ctx.preferred_memory {
            s += 1000;
        }
        if ctx.prefer_hardware && self.acceleration == Acceleration::Hardware {
            s += 100;
        }
        s + self.rank as i32
    }
}

/// What the auto-plug search optimizes for among elements that satisfy the same
/// caps. The default (`System` memory, no hardware preference) reproduces a plain
/// pipeline's selection, so existing behavior is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SelectionContext {
    /// The memory domain the consumer wants the frames in (avoids a download when
    /// it matches a producer's [`CapabilityDescriptor::output_memory`]).
    pub preferred_memory: MemoryDomainKind,
    /// Prefer a hardware path when one is available.
    pub prefer_hardware: bool,
}

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
    /// Auto-plug selection signals (output memory, hardware, rank): how the
    /// search picks this among elements satisfying the same caps. See
    /// [`CapabilityDescriptor`].
    pub capabilities: CapabilityDescriptor,
}

impl ElementDesc {
    /// Build a descriptor from a name and its pad templates. The capabilities
    /// default (software, `System` memory, rank 0); tag a GPU / hardware producer
    /// with the builders below.
    pub fn new(name: &'static str, templates: Vec<PadTemplate>) -> Self {
        Self {
            name,
            templates,
            capabilities: CapabilityDescriptor::default(),
        }
    }

    /// Tag the memory feature this element's source pad produces. Builder form.
    pub fn with_output_memory(mut self, kind: MemoryDomainKind) -> Self {
        self.capabilities.output_memory = kind;
        self
    }

    /// Tag this element's path as hardware-accelerated. Builder form.
    pub fn with_acceleration(mut self, acceleration: Acceleration) -> Self {
        self.capabilities.acceleration = acceleration;
        self
    }

    /// Set the deterministic tiebreaker rank (higher wins among equal scores).
    pub fn with_rank(mut self, rank: i16) -> Self {
        self.capabilities.rank = rank;
        self
    }

    /// The memory feature the element's source pad emits.
    pub fn output_memory(&self) -> MemoryDomainKind {
        self.capabilities.output_memory
    }

    /// First sink (input) pad template, if any.
    fn sink(&self) -> Option<&PadTemplate> {
        self.templates
            .iter()
            .find(|t| t.direction == PadDirection::Sink)
    }

    /// First source (output) pad template, if any.
    fn source(&self) -> Option<&PadTemplate> {
        self.templates
            .iter()
            .find(|t| t.direction == PadDirection::Source)
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
    matches!(
        caps,
        Caps::Audio {
            format: AudioFormat::PcmS16Le | AudioFormat::PcmF32Le,
            ..
        }
    )
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
    find_chain_preferring(descs, input, target, max_depth, MemoryDomainKind::System)
}

/// Like [`find_chain`], but biases ties toward a chain whose terminal element
/// emits the `preferred` memory feature. Thin wrapper over [`find_chain_with`]
/// with a memory-only [`SelectionContext`].
pub fn find_chain_preferring(
    descs: &[ElementDesc],
    input: &Caps,
    target: &dyn Fn(&Caps) -> bool,
    max_depth: usize,
    preferred: MemoryDomainKind,
) -> Option<Vec<ChainLink>> {
    find_chain_with(
        descs,
        input,
        target,
        max_depth,
        SelectionContext {
            preferred_memory: preferred,
            prefer_hardware: false,
        },
    )
}

/// Like [`find_chain`], but scores candidates against `ctx` ([`SelectionContext`])
/// to choose among elements that satisfy the same caps. The search is still
/// breadth-first, so a shorter chain always wins; among equal-length chains, the
/// higher-scoring candidate ([`CapabilityDescriptor::score`]) is tried first, so a
/// GPU consumer can request `Cuda` and get `NvDec` ahead of a CPU decoder.
///
/// The context only reorders which candidate is *tried first* at each hop; if no
/// chain matching the preference exists, the search still finds any valid one. A
/// default `ctx` scores every candidate equally, so the visit order is
/// registration order and a plain pipeline's selection is unchanged.
pub fn find_chain_with(
    descs: &[ElementDesc],
    input: &Caps,
    target: &dyn Fn(&Caps) -> bool,
    max_depth: usize,
    ctx: SelectionContext,
) -> Option<Vec<ChainLink>> {
    if target(input) {
        return Some(Vec::new());
    }
    // Visit order: highest score first, ties broken by registration order (the
    // sort is stable). A default context scores all candidates 0, so this is
    // registration order (no behavior change).
    let mut order: Vec<usize> = (0..descs.len()).collect();
    order.sort_by_key(|&i| core::cmp::Reverse(descs[i].capabilities.score(&ctx)));
    // BFS frontier: each entry is a reached caps state and the element path
    // that produced it. Depth is bounded by max_depth so an unsatisfiable
    // target terminates even with cycle-free same-shape elements.
    let mut frontier: Vec<(Caps, Vec<ChainLink>)> = Vec::from([(input.clone(), Vec::new())]);
    for _ in 0..max_depth {
        let mut next: Vec<(Caps, Vec<ChainLink>)> = Vec::new();
        for (caps, path) in &frontier {
            for &i in &order {
                if path.iter().any(|link| link.index == i) {
                    continue;
                }
                let desc = &descs[i];
                let Some(out_set) = desc.step(caps) else {
                    continue;
                };
                for out in out_set.alternatives() {
                    let mut new_path = path.clone();
                    new_path.push(ChainLink {
                        index: i,
                        output: out.clone(),
                    });
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

/// The media kind a demux output-pad / muxer input-pad reference names (M476,
/// input side M481): `video_0` -> [`Video`](PadKind::Video), `audio_1` ->
/// [`Audio`](PadKind::Audio), a bare `d.` / `m.` or `src_2` / `sink_2` ->
/// [`Any`](PadKind::Any). Defined outside the std-gated `factory` module because
/// the `no_std` fan-in trait ([`MultiInputElement::input_pad_index`]) uses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadKind {
    Video,
    Audio,
    Text,
    Any,
}

/// A parsed pad reference from a `gst-launch` line (M476 output / M481 input),
/// e.g. `d.video_0` -> `{ Video, 0 }`, `m.audio_1` -> `{ Audio, 1 }`, a bare `d.`
/// -> `{ Any, n }`. Handed to a [`DemuxSelectHook`] (output) or
/// [`MultiInputElement::input_pad_index`] (input) to map each request to a stream
/// or input index.
///
/// [`MultiInputElement::input_pad_index`]: crate::MultiInputElement::input_pad_index
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadRequest {
    pub kind: PadKind,
    pub index: usize,
}

#[cfg(feature = "std")]
mod factory {
    use super::*;
    use alloc::boxed::Box;

    use alloc::format;
    use alloc::string::{String, ToString};

    use crate::element::{AsyncElement, DynAsyncElement};
    use crate::fanout::MultiOutputElement;
    use crate::graph::{Graph, GraphError, NodeId, PadId};
    use crate::pad_template::{PadCaps, PadDirection, PadTemplate, PadTemplates};
    use crate::property::format_specs;
    use crate::runtime::launch::ParseError;
    use crate::runtime::{
        DynMultiInputElement, DynMultiOutputElement, DynSourceLoop, GraphNode, GraphNodeRef,
    };

    /// Structured, owned introspection of one registered element: the same facts
    /// [`Registry::inspect`] renders as a `gst-inspect` text dump, exposed as data
    /// so external tooling (the searchable element reference on the docs site,
    /// generated by `g2g-docgen`) reads from one source of truth. Built by
    /// [`Registry::describe`] / [`Registry::describe_all`].
    #[derive(Debug, Clone)]
    pub struct ElementDoc {
        /// Registry / `gst-launch` name.
        pub name: String,
        /// Human-readable long name (may be empty).
        pub long_name: String,
        /// Classification (`klass`), e.g. `"Codec/Encoder/Audio"`.
        pub klass: String,
        /// One-paragraph description of what the element does.
        pub description: String,
        /// Author / origin.
        pub author: String,
        /// `"source"`, `"element"`, or `"muxer (fan-in)"`.
        pub role: String,
        /// Output caps (sources / muxers), the debug spelling; `None` for a
        /// transform / sink, which advertises pad templates instead.
        pub caps: Option<String>,
        /// Pad templates as `"DIR: caps"` lines (transforms / sinks).
        pub pads: Vec<String>,
        /// Settable properties, in declaration order.
        pub properties: Vec<PropertyDoc>,
    }

    /// One settable property, owned (the [`ElementDoc`] counterpart of a
    /// [`PropertySpec`](crate::PropertySpec)).
    #[derive(Debug, Clone)]
    pub struct PropertyDoc {
        /// Property name (`key=value` in a launch line).
        pub name: String,
        /// One-line description.
        pub blurb: String,
        /// Type label as `gst-inspect` names it (`"String"`, `"Unsigned Integer"`).
        pub type_label: String,
        /// Default value as text, if any.
        pub default: Option<String>,
        /// Accepted `(min, max)` range as text, for a numeric property.
        pub range: Option<(String, String)>,
        /// Named choices of an enum-like string property.
        pub enum_values: Option<String>,
        pub readable: bool,
        pub writable: bool,
    }

    /// Owned [`PropertyDoc`]s from a static spec table (shared by `describe`'s
    /// three element-kind branches).
    fn property_docs(specs: &[crate::property::PropertySpec]) -> Vec<PropertyDoc> {
        specs
            .iter()
            .map(|s| PropertyDoc {
                name: s.name.to_string(),
                blurb: s.blurb.to_string(),
                type_label: s.kind.label().to_string(),
                default: s.default.map(|d| d.to_string()),
                range: s.range.map(|(a, b)| (a.to_string(), b.to_string())),
                enum_values: s.enum_values.map(|v| v.to_string()),
                readable: s.flags.readable,
                writable: s.flags.writable,
            })
            .collect()
    }

    /// Pad templates as `"DIR: caps"` lines, matching `format_templates`.
    fn pad_docs(templates: &[PadTemplate]) -> Vec<String> {
        templates
            .iter()
            .map(|t| {
                let dir = match t.direction {
                    PadDirection::Sink => "SINK",
                    PadDirection::Source => "SRC",
                };
                match &t.caps {
                    PadCaps::Fixed(set) => format!("{dir}: {:?}", set.alternatives()),
                    PadCaps::Any => format!("{dir}: ANY"),
                }
            })
            .collect()
    }

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
            Self {
                desc: ElementDesc::new(name, templates),
                build,
            }
        }

        /// Build from a [`PadTemplates`] type, pulling its templates from the
        /// trait so the registration site names only the type and constructor.
        pub fn of<E: PadTemplates>(
            name: &'static str,
            build: fn(&Caps) -> Box<dyn DynAsyncElement>,
        ) -> Self {
            Self::new(name, E::pad_templates(), build)
        }

        /// Tag the memory feature this factory's element produces on its source
        /// pad (see [`ElementDesc::output_memory`]); used by the domain-aware
        /// auto-plug search to prefer e.g. a `Cuda` producer for a GPU consumer.
        /// Builder form, defaulting to `System`.
        pub fn produces(mut self, kind: crate::memory::MemoryDomainKind) -> Self {
            self.desc = self.desc.with_output_memory(kind);
            self
        }

        /// Tag this factory's element as a hardware-accelerated path, so a
        /// hardware-preferring auto-plug search ([`SelectionContext::prefer_hardware`])
        /// favors it. Builder form.
        pub fn hardware(mut self) -> Self {
            self.desc = self.desc.with_acceleration(Acceleration::Hardware);
            self
        }

        /// Set the deterministic tiebreaker rank (higher wins among candidates of
        /// equal score); the explicit-override knob. Builder form, default 0.
        pub fn rank(mut self, rank: i16) -> Self {
            self.desc = self.desc.with_rank(rank);
            self
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
            Self {
                name,
                templates,
                build,
            }
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
            f.debug_struct("LaunchFactory")
                .field("name", &self.name)
                .finish_non_exhaustive()
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
            f.debug_struct("MuxerFactory")
                .field("name", &self.name)
                .finish_non_exhaustive()
        }
    }

    /// A named fan-out demuxer factory for the `gst-launch` parser (M210): the
    /// transpose of [`MuxerFactory`]. A 1-to-N element built per use with the
    /// output count the parser derives from link degree (the `d.` references),
    /// because a [`MultiOutputElement`](crate::fanout::MultiOutputElement)'s
    /// output-pad count must match the demux node's fan-out.
    pub struct DemuxFactory {
        name: &'static str,
        build: fn(usize) -> Box<dyn DynMultiOutputElement>,
    }

    /// Factory for a terminal fan-out *source* (M727): a 0-in / N-out
    /// [`MultiOutputSource`](crate::fanout::MultiOutputSource) generating every
    /// output itself (a WebRTC session receiving its tracks). The build takes
    /// the linked output count so the parser can validate it against the
    /// element's intrinsic port count.
    pub struct FanoutSrcFactory {
        name: &'static str,
        build: fn(usize) -> Box<dyn crate::fanout::DynMultiOutputSource>,
    }

    impl FanoutSrcFactory {
        pub fn new(
            name: &'static str,
            build: fn(usize) -> Box<dyn crate::fanout::DynMultiOutputSource>,
        ) -> Self {
            Self { name, build }
        }

        /// This factory's element name.
        pub fn name(&self) -> &'static str {
            self.name
        }
    }

    impl core::fmt::Debug for FanoutSrcFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("FanoutSrcFactory")
                .field("name", &self.name)
                .finish_non_exhaustive()
        }
    }

    impl DemuxFactory {
        /// Register a fan-out demuxer by name and an output-count constructor
        /// (`|n| Box::new(MyDemux::new(n, ...))`).
        pub fn new(name: &'static str, build: fn(usize) -> Box<dyn DynMultiOutputElement>) -> Self {
            Self { name, build }
        }

        /// This factory's element name.
        pub fn name(&self) -> &'static str {
            self.name
        }
    }

    impl core::fmt::Debug for DemuxFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("DemuxFactory")
                .field("name", &self.name)
                .finish_non_exhaustive()
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
        pub fn new(
            name: &'static str,
            output: Caps,
            build: fn() -> Box<dyn DynSourceLoop>,
        ) -> Self {
            Self {
                name,
                output,
                build,
            }
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
            f.debug_struct("UriSourceFactory")
                .field("scheme", &self.scheme)
                .finish_non_exhaustive()
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

    /// One output branch of a `playbin` graph (M379): the elementary stream a
    /// demux port carries, the raw target to decode it to, and the sink it ends in.
    /// The caller derives these from the demux's announced
    /// [`StreamCollection`](crate::stream::StreamCollection) and its selection: one
    /// `PlaybinPort` per selected stream, in demux port order.
    pub struct PlaybinPort {
        /// The port's elementary-stream caps (e.g. H.264), the decode-chain input
        /// the registry auto-plugs from.
        pub input_caps: Caps,
        /// The raw shape to decode the port to (commonly [`is_raw_video`] for a
        /// video port, [`is_raw_audio`] for an audio port).
        pub target: Box<dyn Fn(&Caps) -> bool>,
        /// The terminal sink for this branch (e.g. an `autovideosink` /
        /// `autoaudiosink` chosen by the stream kind).
        pub sink: Box<dyn DynAsyncElement>,
    }

    impl core::fmt::Debug for PlaybinPort {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("PlaybinPort")
                .field("input_caps", &self.input_caps)
                .finish_non_exhaustive()
        }
    }

    /// Why [`Registry::build_playbin_graph`] could not assemble a graph.
    #[derive(Debug)]
    pub enum PlaybinGraphError {
        /// No output ports were given (a `playbin` needs at least one stream).
        NoPorts,
        /// The URI could not be dispatched to a source (wraps the scheme failure).
        Uri(UriError),
        /// Linking the source to the demux failed.
        Graph(GraphError),
        /// A port's decode chain could not be spliced (no chain, or a link error).
        Decode(DecodebinError),
    }

    impl From<UriError> for PlaybinGraphError {
        fn from(e: UriError) -> Self {
            PlaybinGraphError::Uri(e)
        }
    }
    impl From<GraphError> for PlaybinGraphError {
        fn from(e: GraphError) -> Self {
            PlaybinGraphError::Graph(e)
        }
    }
    impl From<DecodebinError> for PlaybinGraphError {
        fn from(e: DecodebinError) -> Self {
            PlaybinGraphError::Decode(e)
        }
    }

    /// A `playbin uri=X` auto-fan-out hook (M382): given the registry and the
    /// URI, probe the container and assemble a complete multi-stream
    /// `source -> demux -> per-stream decode -> auto sink` graph, the auto
    /// counterpart of [`Registry::build_playbin_graph`]. `Ok(Some(graph))`
    /// handled it; `Ok(None)` declined (e.g. an unprobed scheme or a container
    /// the hook does not parse), so [`parse_launch`](crate::runtime::parse_launch)
    /// falls back to single-stream `playbin`; `Err` aborts the parse. A plain
    /// `fn` pointer, so `Registry` stays `Default` / `Debug`; the plugin crate
    /// that owns the container parsing registers it via
    /// [`Registry::register_playbin`]. Cross-crate by design: the text DSL lives
    /// in core, the Matroska parsing in `g2g-plugins`.
    pub type PlaybinHook = fn(&Registry, &str) -> Result<Option<Graph<GraphNode>>, ParseError>;

    /// An explicit-demux fan-out hook (M476), the sibling of [`PlaybinHook`] for a
    /// named demux element inside a user-authored line
    /// (`filesrc location=x.mkv ! matroskademux name=d  d.video_0 ! ...  d.audio_0 ! ...`).
    /// Given the demux element name, its upstream file location, and the ordered
    /// output-pad requests the line makes, the hook probes the file and builds the
    /// multi-output demuxer with one port per request (in request order), or
    /// returns `None` to decline (a different hook's container, an unreadable file,
    /// or an unsatisfiable request). Cross-crate by design: the DSL lives in core,
    /// the container probing in `g2g-plugins`. Registered via
    /// [`Registry::register_demux_select`].
    pub type DemuxSelectHook = fn(
        name: &str,
        location: &str,
        pads: &[PadRequest],
    ) -> Option<Box<dyn DynMultiOutputElement>>;

    /// A `decodebin` fan-out hook (M482): the decode-per-port sibling of
    /// [`DemuxSelectHook`], for `filesrc location=x.mkv ! decodebin name=d
    /// d.video_0 ! ...  d.audio_0 ! ...`. Unlike the demux-select hook it is NOT
    /// keyed by an element name (a `decodebin` names no container): each registered
    /// hook probes the file and returns `Some` only for the container it parses, so
    /// the parser tries them in turn. It returns the multi-output demuxer plus each
    /// selected port's elementary caps, so the parser can auto-plug a decoder chain
    /// onto every port (the demuxer emits elementary streams; `decodebin` decodes
    /// them to raw). `None` declines (a different container, an unreadable file, or
    /// an unsatisfiable request). Registered via [`Registry::register_decodebin_select`].
    pub type DecodebinSelectHook = fn(
        location: &str,
        pads: &[PadRequest],
    ) -> Option<(Box<dyn DynMultiOutputElement>, Vec<Caps>)>;

    /// The single-stream demux + decode chain a bare (non-fan-out) `decodebin`
    /// should build for a file-backed container whose default single-stream demux
    /// guesses the wrong port (M746). A single-stream demuxer fixes its output pad
    /// at negotiation before parsing any byte, so it defaults to a video port; an
    /// audio-only MPEG-TS then auto-plugs a video decoder and fails "no caps
    /// overlap". `expand_decodebin` consults a hook so the demux instead selects the
    /// container's real (audio) stream, building `filesrc ! demux stream=X !
    /// <decoder> ! ...`.
    #[derive(Debug, Clone)]
    pub struct PrimaryStream {
        /// The single-stream demux element to plug (e.g. `"tsdemux"`).
        pub demux: &'static str,
        /// Properties that select the primary stream on `demux` (e.g.
        /// `[("stream", "aac")]`).
        pub props: Vec<(String, String)>,
        /// The elementary caps the demux emits for that stream, the auto-plug input
        /// for the decoder chain spliced after it.
        pub caps: Caps,
    }

    /// A bare-`decodebin` primary-stream hook (M746): given the upstream file
    /// location and the container caps, sniff the container and return the
    /// single-stream demux + stream selection for its primary decodable stream, or
    /// `None` when the default video port is correct (a video track is present), the
    /// file is unreadable, or the hook does not parse the container. Registered via
    /// [`Registry::register_primary_stream`].
    pub type PrimaryStreamHook = fn(location: &str, caps: &Caps) -> Option<PrimaryStream>;

    impl core::fmt::Debug for ElementFactory {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("ElementFactory")
                .field("name", &self.desc.name)
                .finish_non_exhaustive()
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
        demuxes: Vec<DemuxFactory>,
        fanout_srcs: Vec<FanoutSrcFactory>,
        /// gst-canonical-name aliases (M192): each maps a name to an ordered list
        /// of registered targets, the first that is actually registered wins. A
        /// plain rename is a one-entry list; `autovideosink` is a fallback chain
        /// (`waylandsink`, `kmssink`, ..., `fakesink`). Resolved at `make_*` time,
        /// so an alias whose targets are all feature-gated-out simply misses.
        aliases: Vec<(&'static str, &'static [&'static str])>,
        /// The `playbin uri=X` auto-fan-out hooks (M382, multi-hook M389). A lone
        /// `playbin` in a text pipeline tries each in registration order until one
        /// handles the URI (`Ok(Some)`); each declines (`Ok(None)`) a container it
        /// does not parse, so one hook per container type coexists (MKV, TS, ...).
        /// Empty (the default) leaves `playbin` as the M196 single-stream pipeline.
        playbin: Vec<PlaybinHook>,
        /// Explicit-demux fan-out hooks (M476): a named demux element with several
        /// output-pad references and a file source upstream tries each in order
        /// until one builds the multi-output demuxer (`Some`); each declines
        /// (`None`) a container it does not parse. Empty (the default) leaves a
        /// demux element single-stream (its M114/M180 launch registration).
        demux_select: Vec<DemuxSelectHook>,
        /// `decodebin` fan-out hooks (M482): the decode-per-port sibling of
        /// `demux_select`. A `decodebin name=d` with several `d.` refs and a file
        /// upstream tries each until one parses the container, returning the
        /// demuxer + per-port caps so the parser splices a decoder onto each port.
        decodebin_select: Vec<DecodebinSelectHook>,
        /// Bare-`decodebin` primary-stream hooks (M746): a `filesrc location=X !
        /// decodebin` on a container tries each until one sniffs the file and names
        /// the single-stream demux + stream selection for its primary decodable
        /// stream. Empty (the default) keeps the demux's declared default port, so
        /// an audio-only container plugs the wrong (video) decoder.
        primary_stream: Vec<PrimaryStreamHook>,
        /// Optional parser injector (M421): given a decode-chain input caps,
        /// returns a parser element to splice in just before the decoder (e.g. an
        /// access-unit-re-framing `h264parse` for `CompressedVideo { H264, .. }`),
        /// or `None` to decode the input directly. A decoder fed un-aligned units
        /// (one MPEG-TS PES that is not one access unit) mis-parses; this is how
        /// the registry mirrors GStreamer's `decodebin` always inserting a parser.
        /// Maps input caps to the LAUNCH NAME of the parser (M676: a name, not a
        /// boxed element, so the name-based `decodebin` expansion in `parse_launch`
        /// shares the one mapping); the element is built via
        /// [`make_element`](Self::make_element). A bare function pointer so the
        /// registry stays `Debug` + `Default`; [`default_registry`](crate) sets it.
        /// `None` (the default) decodes directly, preserving the prior behaviour
        /// for registries that do not set it.
        parser_provider: Option<fn(&Caps) -> Option<&'static str>>,
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

        /// Register a named fan-out demuxer for the `gst-launch` parser (M210),
        /// returning `&mut self` to chain calls.
        pub fn register_fanout_src(&mut self, factory: FanoutSrcFactory) -> &mut Self {
            self.fanout_srcs.push(factory);
            self
        }

        /// Whether `name` is a registered terminal fan-out source.
        pub fn is_fanout_src(&self, name: &str) -> bool {
            self.fanout_srcs.iter().any(|f| f.name == name)
        }

        /// Build a registered terminal fan-out source with `outputs` ports.
        pub fn make_fanout_src(
            &self,
            name: &str,
            outputs: usize,
        ) -> Option<Box<dyn crate::fanout::DynMultiOutputSource>> {
            self.fanout_srcs
                .iter()
                .find(|f| f.name == name)
                .map(|f| (f.build)(outputs))
        }

        pub fn register_demux(&mut self, factory: DemuxFactory) -> &mut Self {
            self.demuxes.push(factory);
            self
        }

        /// Register a `playbin uri=X` auto-fan-out hook (M382): a lone `playbin`
        /// in a [`parse_launch`](crate::runtime::parse_launch) pipeline tries the
        /// registered hooks in order until one handles the URI. Register one per
        /// container type (MKV, TS, ...); each declines a container it does not
        /// parse. Returns `&mut self` to chain calls.
        pub fn register_playbin(&mut self, hook: PlaybinHook) -> &mut Self {
            self.playbin.push(hook);
            self
        }

        /// The registered explicit-demux fan-out hooks (M476), tried in order by
        /// [`parse_launch`](crate::runtime::parse_launch) for a named demux element
        /// with several output-pad references and a file source upstream.
        pub fn demux_select_hooks(&self) -> &[DemuxSelectHook] {
            &self.demux_select
        }

        /// Register an explicit-demux fan-out hook (M476): a named demux element
        /// (`matroskademux name=d  d.video_0 ! ...  d.audio_0 ! ...`) fed by a file
        /// source tries the registered hooks in order until one probes the file and
        /// builds the multi-output demuxer. Register one per container type; each
        /// declines a container it does not parse. Returns `&mut self` to chain.
        pub fn register_demux_select(&mut self, hook: DemuxSelectHook) -> &mut Self {
            self.demux_select.push(hook);
            self
        }

        /// The `decodebin` fan-out hooks (M482), tried in order by
        /// [`parse_launch`](crate::runtime::parse_launch) for a `decodebin name=d`
        /// with several `d.` references and a file source upstream.
        pub fn decodebin_select_hooks(&self) -> &[DecodebinSelectHook] {
            &self.decodebin_select
        }

        /// Register a `decodebin` fan-out hook (M482): a `decodebin name=d` fed by a
        /// file source tries the registered hooks until one parses the container,
        /// returning the multi-output demuxer + per-port caps so the parser splices
        /// a decoder onto each port. One per container type. Returns `&mut self`.
        pub fn register_decodebin_select(&mut self, hook: DecodebinSelectHook) -> &mut Self {
            self.decodebin_select.push(hook);
            self
        }

        /// Register a bare-`decodebin` primary-stream hook (M746): a `filesrc
        /// location=X ! decodebin` on a container tries each until one sniffs the
        /// file and names the single-stream demux + stream selection for its primary
        /// decodable stream. One per container type. Returns `&mut self`.
        pub fn register_primary_stream(&mut self, hook: PrimaryStreamHook) -> &mut Self {
            self.primary_stream.push(hook);
            self
        }

        /// The single-stream demux + decode selection for a bare `decodebin` on the
        /// container at `location` with `caps`, from the first hook that parses it
        /// (M746); `None` when none applies (no video-less container to fix, an
        /// unreadable file, or no hook registered).
        pub fn primary_stream(&self, location: &str, caps: &Caps) -> Option<PrimaryStream> {
            self.primary_stream
                .iter()
                .find_map(|hook| hook(location, caps))
        }

        /// Set the parser injector consulted by [`decodebin`](Self::decodebin) and
        /// [`decodebin_preferring`](Self::decodebin_preferring) (M421): before a
        /// decode chain is spliced, `provider(input)` may name a registered parser
        /// (e.g. an access-unit-re-framing `h264parse`) to prepend ahead of the
        /// decoder, so the decoder is fed one access unit per packet. The
        /// name-based `decodebin` expansion in `parse_launch` consults the same
        /// mapping via [`parser_name`](Self::parser_name) (M676). Returns
        /// `&mut self` to chain calls.
        pub fn set_parser_provider(
            &mut self,
            provider: fn(&Caps) -> Option<&'static str>,
        ) -> &mut Self {
            self.parser_provider = Some(provider);
            self
        }

        /// The launch name of the re-framing parser `decodebin` prepends ahead of
        /// a decoder for `input` caps, if the provider names one (M676).
        pub fn parser_name(&self, input: &Caps) -> Option<&'static str> {
            self.parser_provider.and_then(|provider| provider(input))
        }

        /// Prepend the [`parser_provider`](Self::set_parser_provider)'s parser to a
        /// just-autoplugged decode chain, when one is configured and the chain is a
        /// real decode (non-empty: an already-satisfied input needs no parser).
        fn maybe_prepend_parser(&self, input: &Caps, elements: &mut Vec<Box<dyn DynAsyncElement>>) {
            if elements.is_empty() {
                return;
            }
            if let Some(parser) = self
                .parser_name(input)
                .and_then(|name| self.make_element(name))
            {
                elements.insert(0, parser);
            }
        }

        /// The registered [`PlaybinHook`]s, in registration order (empty if none,
        /// so `playbin` stays the M196 single-stream pipeline). The parser tries
        /// them in turn for a lone `playbin uri=`.
        pub fn playbin_hooks(&self) -> &[PlaybinHook] {
            &self.playbin
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
                        || self.muxers.iter().any(|m| m.name == t)
                        || self.demuxes.iter().any(|d| d.name == t)
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
            self.sources
                .iter()
                .find(|s| s.name == name)
                .map(|s| (s.build)())
        }

        /// Construct a registered transform / sink by name (a parser interior or
        /// tail element), default-configured. `None` if `name` is not registered
        /// via [`register_launch`](Self::register_launch) (after alias resolution).
        pub fn make_element(&self, name: &str) -> Option<Box<dyn DynAsyncElement>> {
            let name = self.resolve_alias(name);
            self.launch
                .iter()
                .find(|f| f.name == name)
                .map(|f| (f.build)())
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
            let t = f
                .templates
                .iter()
                .find(|t| t.direction == PadDirection::Source)?;
            match &t.caps {
                PadCaps::Fixed(set) => set.alternatives().first().cloned(),
                PadCaps::Any => None,
            }
        }

        /// Construct a registered fan-in muxer by name with `inputs` input pads
        /// (the parser derives the count from link degree, so it matches the
        /// muxer node's input-pad count). `None` if `name` is not registered via
        /// [`register_muxer`](Self::register_muxer).
        pub fn make_muxer(
            &self,
            name: &str,
            inputs: usize,
        ) -> Option<Box<dyn DynMultiInputElement>> {
            let name = self.resolve_alias(name);
            self.muxers
                .iter()
                .find(|m| m.name == name)
                .map(|m| (m.build)(inputs))
        }

        /// Construct a registered fan-out demuxer by name with `outputs` output
        /// pads (the parser derives the count from the `d.` link degree, so it
        /// matches the demux node's fan-out). `None` if `name` is not registered
        /// via [`register_demux`](Self::register_demux).
        pub fn make_demux(
            &self,
            name: &str,
            outputs: usize,
        ) -> Option<Box<dyn DynMultiOutputElement>> {
            let name = self.resolve_alias(name);
            self.demuxes
                .iter()
                .find(|d| d.name == name)
                .map(|d| (d.build)(outputs))
        }

        /// Whether `name` is registered as a fan-out demuxer (the parser uses this
        /// to allow a multi-output node without an explicit `tee`).
        pub fn is_demux(&self, name: &str) -> bool {
            self.demuxes.iter().any(|d| d.name == name)
        }

        /// The names of every element registerable by the parser: sources first,
        /// then transforms / sinks, each in registration order. The `gst-inspect`
        /// element list.
        pub fn element_names(&self) -> Vec<&'static str> {
            self.sources
                .iter()
                .map(|s| s.name)
                .chain(self.launch.iter().map(|f| f.name))
                // A muxer dual-registered as a launch element (e.g. `mp4mux`) is
                // already listed above; only emit muxer-only names here.
                .chain(
                    self.muxers
                        .iter()
                        .map(|m| m.name)
                        .filter(|name| !self.launch.iter().any(|f| f.name == *name)),
                )
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
                // Skip a muxer already listed as a launch element above.
                if !self.launch.iter().any(|f| f.name == m.name) {
                    lines.push(m.name.to_string());
                }
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
            use crate::property::format_metadata;
            use core::fmt::Write;
            let mut out = String::new();
            if let Some(s) = self.sources.iter().find(|s| s.name == name) {
                let src = (s.build)();
                out.push_str(&format_metadata(name, &src.metadata()));
                let _ = writeln!(out, "  Role        source");
                let _ = writeln!(out, "\nOutput caps:\n  {:?}", s.output);
                let _ = write!(
                    out,
                    "\nElement Properties:\n{}",
                    format_specs(src.properties())
                );
                Some(out)
            } else if let Some(f) = self.launch.iter().find(|f| f.name == name) {
                let el = (f.build)();
                out.push_str(&format_metadata(name, &el.metadata()));
                let _ = writeln!(out, "  Role        element");
                let _ = write!(out, "\nPad Templates:\n{}", format_templates(&f.templates));
                let _ = write!(
                    out,
                    "\nElement Properties:\n{}",
                    format_specs(el.properties())
                );
                Some(out)
            } else if let Some(m) = self.muxers.iter().find(|m| m.name == name) {
                // Build a one-input instance to read its metadata / properties (the
                // specs are `&'static`, behind instance methods); the fan-in
                // constructors are side-effect-free, so this is safe. The real input
                // count is derived from link degree at parse, not needed here.
                let mux = (m.build)(1);
                out.push_str(&format_metadata(name, &mux.metadata()));
                let _ = writeln!(out, "  Role        muxer (fan-in)");
                if let Ok(caps) = mux.output_caps() {
                    let _ = writeln!(out, "\nOutput caps:\n  {caps:?}");
                }
                let _ = writeln!(out, "\nInputs: derived from link degree");
                let _ = write!(
                    out,
                    "\nElement Properties:\n{}",
                    format_specs(mux.properties())
                );
                Some(out)
            } else {
                None
            }
        }

        /// Structured introspection: the same facts [`inspect`](Self::inspect)
        /// dumps as text, returned as an [`ElementDoc`] for tooling to render
        /// (e.g. the searchable element reference the docs site is generated from).
        /// `None` if the name is not registered. Like `inspect`, it default-builds
        /// the element to read its `&'static` metadata / property table, so the
        /// in-tree constructors must be side-effect-free.
        pub fn describe(&self, name: &str) -> Option<ElementDoc> {
            if let Some(s) = self.sources.iter().find(|s| s.name == name) {
                let src = (s.build)();
                let m = src.metadata();
                Some(ElementDoc {
                    name: name.to_string(),
                    long_name: m.long_name.to_string(),
                    klass: m.klass.to_string(),
                    description: m.description.to_string(),
                    author: m.author.to_string(),
                    role: "source".to_string(),
                    caps: Some(format!("{:?}", s.output)),
                    pads: Vec::new(),
                    properties: property_docs(src.properties()),
                })
            } else if let Some(f) = self.launch.iter().find(|f| f.name == name) {
                let el = (f.build)();
                let m = el.metadata();
                Some(ElementDoc {
                    name: name.to_string(),
                    long_name: m.long_name.to_string(),
                    klass: m.klass.to_string(),
                    description: m.description.to_string(),
                    author: m.author.to_string(),
                    role: "element".to_string(),
                    caps: None,
                    pads: pad_docs(&f.templates),
                    properties: property_docs(el.properties()),
                })
            } else if let Some(mx) = self.muxers.iter().find(|m| m.name == name) {
                let mux = (mx.build)(1);
                let m = mux.metadata();
                Some(ElementDoc {
                    name: name.to_string(),
                    long_name: m.long_name.to_string(),
                    klass: m.klass.to_string(),
                    description: m.description.to_string(),
                    author: m.author.to_string(),
                    role: "muxer (fan-in)".to_string(),
                    caps: mux.output_caps().ok().map(|c| format!("{c:?}")),
                    pads: Vec::new(),
                    properties: property_docs(mux.properties()),
                })
            } else {
                None
            }
        }

        /// Every registered element as an [`ElementDoc`], in the same order as
        /// [`element_names`](Self::element_names) (sources, transforms / sinks,
        /// then muxer-only names). The structured catalog behind `g2g-docgen`.
        pub fn describe_all(&self) -> Vec<ElementDoc> {
            self.element_names()
                .into_iter()
                .filter_map(|n| self.describe(n))
                .collect()
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
            Some(
                chain
                    .into_iter()
                    .map(|link| self.factories[link.index].desc.name)
                    .collect(),
            )
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
            let mut elements = self
                .autoplug(input, target, max_depth)
                .ok_or(DecodebinError::NoChain)?;
            self.maybe_prepend_parser(input, &mut elements);
            Self::splice_chain(graph, from, to, elements)
        }

        /// Insert `elements` as a run of transforms between output pad `from` and
        /// input pad `to`, returning the inserted node ids in chain order. The
        /// shared splice for [`decodebin`](Self::decodebin) and
        /// [`decodebin_preferring`](Self::decodebin_preferring).
        fn splice_chain(
            graph: &mut Graph<GraphNode>,
            from: impl Into<PadId>,
            to: impl Into<PadId>,
            elements: Vec<Box<dyn DynAsyncElement>>,
        ) -> Result<Vec<NodeId>, DecodebinError> {
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

        /// Domain-aware [`autoplug_names`](Self::autoplug_names): bias ties toward a
        /// chain whose terminal element emits the `preferred` memory feature (see
        /// [`ElementDesc::output_memory`]). `MemoryDomainKind::System` reproduces
        /// the default selection; `Cuda` prefers e.g. `NvDec` over a CPU decoder.
        pub fn autoplug_names_preferring(
            &self,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
            preferred: MemoryDomainKind,
        ) -> Option<Vec<&'static str>> {
            let descs = self.descs();
            let chain = find_chain_preferring(&descs, input, target, max_depth, preferred)?;
            Some(
                chain
                    .into_iter()
                    .map(|link| self.factories[link.index].desc.name)
                    .collect(),
            )
        }

        /// Domain-aware [`autoplug`](Self::autoplug): instantiate the chain the
        /// domain-aware search picks (see
        /// [`autoplug_names_preferring`](Self::autoplug_names_preferring)).
        pub fn autoplug_preferring(
            &self,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
            preferred: MemoryDomainKind,
        ) -> Option<Vec<Box<dyn DynAsyncElement>>> {
            let descs = self.descs();
            let chain = find_chain_preferring(&descs, input, target, max_depth, preferred)?;
            Some(
                chain
                    .into_iter()
                    .map(|link| self.factories[link.index].build(&link.output))
                    .collect(),
            )
        }

        /// Capability-aware [`autoplug_names`](Self::autoplug_names): score
        /// candidates against `ctx` ([`SelectionContext`]) to choose among
        /// elements satisfying the same caps (memory domain, hardware, then a rank
        /// tiebreaker). The generalization of
        /// [`autoplug_names_preferring`](Self::autoplug_names_preferring) (which is
        /// the memory-only case); a default `ctx` is the plain selection.
        pub fn autoplug_names_with(
            &self,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
            ctx: SelectionContext,
        ) -> Option<Vec<&'static str>> {
            let descs = self.descs();
            let chain = find_chain_with(&descs, input, target, max_depth, ctx)?;
            Some(
                chain
                    .into_iter()
                    .map(|link| self.factories[link.index].desc.name)
                    .collect(),
            )
        }

        /// Capability-aware [`autoplug`](Self::autoplug): instantiate the chain the
        /// capability-scored search ([`autoplug_names_with`](Self::autoplug_names_with))
        /// picks.
        pub fn autoplug_with(
            &self,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
            ctx: SelectionContext,
        ) -> Option<Vec<Box<dyn DynAsyncElement>>> {
            let descs = self.descs();
            let chain = find_chain_with(&descs, input, target, max_depth, ctx)?;
            Some(
                chain
                    .into_iter()
                    .map(|link| self.factories[link.index].build(&link.output))
                    .collect(),
            )
        }

        /// Domain-aware [`decodebin`](Self::decodebin): splice in the chain the
        /// domain-aware search picks, biased toward `preferred` memory (e.g. `Cuda`
        /// to prefer `NvDec` when the downstream consumer is GPU-resident).
        #[allow(clippy::too_many_arguments)]
        pub fn decodebin_preferring(
            &self,
            graph: &mut Graph<GraphNode>,
            from: impl Into<PadId>,
            to: impl Into<PadId>,
            input: &Caps,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
            preferred: MemoryDomainKind,
        ) -> Result<Vec<NodeId>, DecodebinError> {
            let mut elements = self
                .autoplug_preferring(input, target, max_depth, preferred)
                .ok_or(DecodebinError::NoChain)?;
            self.maybe_prepend_parser(input, &mut elements);
            Self::splice_chain(graph, from, to, elements)
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
        /// Dispatch a URI to its registered scheme handler, constructing the
        /// source from the URI and returning it with the caps it produces (the
        /// decode-chain input). The lower half of
        /// [`build_uridecodebin`](Self::build_uridecodebin), exposed so the
        /// `uridecodebin` / `playbin` text-parser nodes can splice the source into
        /// a larger graph instead of getting a complete one.
        pub fn build_uri_source(
            &self,
            uri: &str,
        ) -> Result<(Box<dyn DynSourceLoop>, Caps), UriError> {
            let parsed = Uri::parse(uri).ok_or(UriError::Malformed)?;
            let handler = self
                .uris
                .iter()
                .find(|h| h.scheme == parsed.scheme)
                .ok_or(UriError::UnknownScheme)?;
            (handler.build)(&parsed)
        }

        pub fn build_uridecodebin<Sk: AsyncElement + 'static>(
            &self,
            uri: &str,
            sink: Sk,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
        ) -> Result<Graph<GraphNode>, UriError> {
            let (source, output) = self.build_uri_source(uri)?;
            self.build_source_decodebin(source, &output, sink, target, max_depth)
        }

        /// Auto-plug `source -> chain -> sink` from an *already constructed*
        /// source whose output is `source_caps`, the lower half of
        /// [`build_uridecodebin`](Self::build_uridecodebin) factored out so a
        /// caller that built (or wrapped) its own source can still get the decode
        /// chain auto-plugged. A gapless playlist source (`GaplessSrc`) uses this
        /// to splice its decode chain without the URI-handler step.
        pub fn build_source_decodebin<Sk: AsyncElement + 'static>(
            &self,
            source: Box<dyn DynSourceLoop>,
            source_caps: &Caps,
            sink: Sk,
            target: &dyn Fn(&Caps) -> bool,
            max_depth: usize,
        ) -> Result<Graph<GraphNode>, UriError> {
            let mut graph: Graph<GraphNode> = Graph::new();
            let src = graph.add_source(GraphNodeRef::Source(source));
            let snk = graph.add_sink(GraphNodeRef::element(sink));
            self.decodebin(&mut graph, src, snk, source_caps, target, max_depth)?;
            Ok(graph)
        }

        /// `playbin`-equivalent (M379): assemble a complete runnable graph that
        /// splits a container into its selected streams and decodes each to its own
        /// sink. Builds the source from `uri`, adds `demux` (a
        /// [`MultiOutputElement`], e.g. `MkvDemuxN`) as a fan-out node, and for each
        /// [`PlaybinPort`] (one per selected stream, in port order) auto-plugs a
        /// decode chain from that port's elementary caps to its sink. Returns
        /// `source -> demux -> {decode chain -> sink}` ready for
        /// [`run_graph`](crate::runtime::run_graph), the multi-stream counterpart of
        /// [`build_uridecodebin`](Self::build_uridecodebin).
        ///
        /// The app derives `ports` from the demux's announced
        /// [`StreamCollection`](crate::stream::StreamCollection) (M376) and its
        /// selection (M377); `demux`'s port count must equal `ports.len()`. Each
        /// branch retypes from the demux's (byte-stream) input caps to its
        /// elementary stream via the per-port `CapsChanged` the demux emits, so a
        /// branch element must tolerate the startup broadcast and re-solve then (the
        /// M210 demux-node contract); per-branch static negotiation against the port
        /// caps is a follow-up.
        pub fn build_playbin_graph<D: MultiOutputElement + 'static>(
            &self,
            uri: &str,
            demux: D,
            ports: Vec<PlaybinPort>,
            max_depth: usize,
        ) -> Result<Graph<GraphNode>, PlaybinGraphError> {
            if ports.is_empty() {
                return Err(PlaybinGraphError::NoPorts);
            }
            let (source, _byte_caps) = self.build_uri_source(uri)?;
            self.build_playbin_graph_with_source(source, demux, ports, max_depth)
        }

        /// Like [`build_playbin_graph`](Self::build_playbin_graph) but with a
        /// pre-built byte source instead of one derived from the URI's scheme
        /// handler. The `playbin uri=` auto-fan-out hook (M382) uses this: having
        /// probed the file to choose the demuxer, it already knows the container,
        /// so it supplies the matching raw-byte source directly rather than the
        /// URI handler's source (which, for `file://`, may self-demux a *different*
        /// container, e.g. MP4). `source` must emit the byte stream `demux`
        /// expects.
        pub fn build_playbin_graph_with_source<D: MultiOutputElement + 'static>(
            &self,
            source: Box<dyn DynSourceLoop>,
            demux: D,
            ports: Vec<PlaybinPort>,
            max_depth: usize,
        ) -> Result<Graph<GraphNode>, PlaybinGraphError> {
            if ports.is_empty() {
                return Err(PlaybinGraphError::NoPorts);
            }
            let mut graph: Graph<GraphNode> = Graph::new();
            let src = graph.add_source(GraphNodeRef::Source(source));
            let outputs = ports.len() as u8;
            let demux = graph.add_demux(GraphNode::demux(demux), outputs);
            graph.link(src, demux.input())?;
            for (i, port) in ports.into_iter().enumerate() {
                let snk = graph.add_sink(GraphNodeRef::Element(port.sink));
                self.decodebin(
                    &mut graph,
                    demux.out(i as u8),
                    snk,
                    &port.input_caps,
                    &*port.target,
                    max_depth,
                )?;
            }
            Ok(graph)
        }
    }
}

#[cfg(feature = "std")]
pub use factory::{
    declared_source_caps, DecodebinError, DecodebinSelectHook, DemuxFactory, DemuxSelectHook,
    ElementDoc, ElementFactory, FanoutSrcFactory, LaunchFactory, MuxerFactory, PlaybinError,
    PlaybinGraphError, PlaybinHook, PlaybinPort, PrimaryStream, PrimaryStreamHook, PropertyDoc,
    Registry, SourceFactory, Uri, UriError, UriSourceFactory,
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
        Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
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

    /// A second H.264 -> NV12 decoder tagged as a Cuda-producing hardware path
    /// (a native NVDEC analog), to exercise capability-based selection.
    fn gpu_decoder() -> ElementDesc {
        ElementDesc::new(
            "nvdec",
            Vec::from([
                PadTemplate::sink(CapsSet::one(h264(Dim::Any))),
                PadTemplate::source(CapsSet::one(raw(RawVideoFormat::Nv12))),
            ]),
        )
        .with_output_memory(MemoryDomainKind::Cuda)
        .with_acceleration(Acceleration::Hardware)
    }

    /// Just the descriptor indices of a found chain, for terse assertions.
    fn indices(chain: &[ChainLink]) -> Vec<usize> {
        chain.iter().map(|l| l.index).collect()
    }

    #[test]
    fn capability_score_ranks_domain_then_hardware_then_rank() {
        let sys = CapabilityDescriptor::default();
        let cuda = CapabilityDescriptor {
            output_memory: MemoryDomainKind::Cuda,
            ..Default::default()
        };
        let ctx_cuda = SelectionContext {
            preferred_memory: MemoryDomainKind::Cuda,
            prefer_hardware: false,
        };
        assert!(
            cuda.score(&ctx_cuda) > sys.score(&ctx_cuda),
            "a memory-domain match dominates"
        );

        let hw = CapabilityDescriptor {
            acceleration: Acceleration::Hardware,
            ..Default::default()
        };
        let ctx_hw = SelectionContext {
            preferred_memory: MemoryDomainKind::System,
            prefer_hardware: true,
        };
        assert!(
            hw.score(&ctx_hw) > sys.score(&ctx_hw),
            "a hardware preference favors hardware"
        );

        let ranked = CapabilityDescriptor {
            rank: 5,
            ..Default::default()
        };
        let plain = SelectionContext::default();
        assert!(
            ranked.score(&plain) > sys.score(&plain),
            "rank breaks an otherwise-equal tie"
        );
    }

    #[test]
    fn default_context_keeps_registration_order() {
        // CPU decoder first, GPU second: a plain search picks the CPU decoder, so
        // a default pipeline's selection is unchanged by the capability machinery.
        let descs = [decoder(), gpu_decoder()];
        let chain = find_chain_with(
            &descs,
            &h264(Dim::Any),
            &is_raw_video,
            4,
            SelectionContext::default(),
        )
        .expect("a decoder reaches raw");
        assert_eq!(
            indices(&chain),
            Vec::from([0usize]),
            "default = registration order (CPU first)"
        );
    }

    #[test]
    fn cuda_preference_selects_the_gpu_decoder() {
        // The same two decoders; a Cuda preference flips the choice to the GPU
        // decoder even though it is registered second (avoids a download).
        let descs = [decoder(), gpu_decoder()];
        let ctx = SelectionContext {
            preferred_memory: MemoryDomainKind::Cuda,
            prefer_hardware: false,
        };
        let chain =
            find_chain_with(&descs, &h264(Dim::Any), &is_raw_video, 4, ctx).expect("reaches raw");
        assert_eq!(
            indices(&chain),
            Vec::from([1usize]),
            "Cuda preference picks the GPU decoder"
        );
    }

    #[test]
    fn rank_breaks_ties_among_equal_candidates() {
        // Two equivalent CPU decoders; the higher-ranked one wins regardless of
        // registration order (the explicit-override tiebreaker).
        let descs = [decoder(), decoder().with_rank(10)];
        let chain = find_chain_with(
            &descs,
            &h264(Dim::Any),
            &is_raw_video,
            4,
            SelectionContext::default(),
        )
        .expect("reaches raw");
        assert_eq!(
            indices(&chain),
            Vec::from([1usize]),
            "higher rank wins the tie"
        );
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
        let chain =
            find_chain(&descs, &raw(RawVideoFormat::Nv12), &is_raw_video, 4).expect("already raw");
        assert!(
            chain.is_empty(),
            "no elements needed when input is already raw"
        );
    }

    #[test]
    fn finds_multi_element_chain_to_a_specific_format() {
        // Target a format only the converter produces, forcing decoder -> convert.
        let descs = [parser(), decoder(), convert()];
        let target = |c: &Caps| {
            matches!(
                c,
                Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    ..
                }
            )
        };
        let chain = find_chain(&descs, &h264(Dim::Any), &target, 4)
            .expect("decode then convert reaches RGBA");
        assert_eq!(
            indices(&chain),
            Vec::from([1usize, 2usize]),
            "decoder then converter"
        );
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
        let target = |c: &Caps| {
            matches!(
                c,
                Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    ..
                }
            )
        };
        assert!(
            find_chain(&descs, &h264(Dim::Any), &target, 1).is_none(),
            "1 hop is too shallow"
        );
        assert!(
            find_chain(&descs, &h264(Dim::Any), &target, 2).is_some(),
            "2 hops suffice"
        );
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
        assert_eq!(
            f.rest, "/home/a/clip.mp4",
            "file:// leaves an absolute path"
        );

        let udp = Uri::parse("udp://0.0.0.0:5004").expect("valid udp uri");
        assert_eq!((udp.scheme, udp.rest), ("udp", "0.0.0.0:5004"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn uri_parse_rejects_malformed() {
        assert!(Uri::parse("notauri").is_none(), "no scheme separator");
        assert!(Uri::parse("://nohost").is_none(), "empty scheme");
    }

    /// A trivial element for graph-shape tests: accepts any caps, never runs. The
    /// registered factory's pad-template descriptor (not this body) drives autoplug.
    #[cfg(feature = "std")]
    #[derive(Debug, Default)]
    struct Dummy;
    #[cfg(feature = "std")]
    impl crate::PadTemplates for Dummy {
        fn pad_templates() -> Vec<PadTemplate> {
            Vec::new()
        }
    }
    #[cfg(feature = "std")]
    impl crate::AsyncElement for Dummy {
        type ProcessFuture<'a> = core::pin::Pin<
            alloc::boxed::Box<dyn core::future::Future<Output = Result<(), crate::G2gError>> + 'a>,
        >;
        fn intercept_caps(&self, c: &Caps) -> Result<Caps, crate::G2gError> {
            Ok(c.clone())
        }
        fn configure_pipeline(
            &mut self,
            _c: &Caps,
        ) -> Result<crate::ConfigureOutcome, crate::G2gError> {
            Ok(crate::ConfigureOutcome::Accepted)
        }
        fn process<'a>(
            &'a mut self,
            _p: crate::PipelinePacket,
            _o: &'a mut dyn crate::OutputSink,
        ) -> Self::ProcessFuture<'a> {
            alloc::boxed::Box::pin(async { Ok(()) })
        }
    }

    /// M421: a configured `parser_provider` prepends its parser to every auto-plugged
    /// decode chain (so a decoder is fed access-unit-aligned input), and only then.
    #[cfg(feature = "std")]
    #[test]
    fn decodebin_inserts_the_parser_provider_before_the_decoder() {
        use crate::runtime::{GraphNode, GraphNodeRef};
        use crate::Graph;

        // An H.264 -> raw "decoder" factory; the element body is irrelevant to the
        // graph shape, so a Dummy stands in.
        fn dec_factory() -> ElementFactory {
            ElementFactory::new(
                "h264dec",
                Vec::from([
                    PadTemplate::sink(CapsSet::one(h264(Dim::Any))),
                    PadTemplate::source(CapsSet::one(raw(RawVideoFormat::Nv12))),
                ]),
                |_| alloc::boxed::Box::new(Dummy),
            )
        }
        let build = |provider: bool| -> usize {
            let mut reg = Registry::new();
            reg.register(dec_factory());
            if provider {
                // The provider names a registered launch element (M676), the
                // identity-caps re-framing parser.
                reg.register_launch(LaunchFactory::new(
                    "auparse",
                    Vec::from([
                        PadTemplate::sink(CapsSet::one(h264(Dim::Any))),
                        PadTemplate::source(CapsSet::one(h264(Dim::Any))),
                    ]),
                    || alloc::boxed::Box::new(Dummy),
                ));
                reg.set_parser_provider(|caps| match caps {
                    Caps::CompressedVideo {
                        codec: VideoCodec::H264,
                        ..
                    } => Some("auparse"),
                    _ => None,
                });
            }
            let mut g: Graph<GraphNode> = Graph::new();
            let head = g.add_transform(GraphNodeRef::element(Dummy));
            let tail = g.add_sink(GraphNodeRef::element(Dummy));
            reg.decodebin(&mut g, head, tail, &h264(Dim::Any), &is_raw_video, 4)
                .expect("h264 -> raw chain")
                .len()
        };
        assert_eq!(
            build(false),
            1,
            "no provider: the decode chain is just the decoder"
        );
        assert_eq!(
            build(true),
            2,
            "provider: a parser is spliced in ahead of the decoder"
        );
    }
}
