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

use crate::caps::{Caps, CapsSet};
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

/// Find the shortest chain of registered element types that converts `input`
/// caps into caps satisfying `target`, returning the indices into `descs` in
/// chain order (upstream first).
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
) -> Option<Vec<usize>> {
    if target(input) {
        return Some(Vec::new());
    }
    // BFS frontier: each entry is a reached caps state and the element path
    // that produced it. Depth is bounded by max_depth so an unsatisfiable
    // target terminates even with cycle-free same-shape elements.
    let mut frontier: Vec<(Caps, Vec<usize>)> = Vec::from([(input.clone(), Vec::new())]);
    for _ in 0..max_depth {
        let mut next: Vec<(Caps, Vec<usize>)> = Vec::new();
        for (caps, path) in &frontier {
            for (i, desc) in descs.iter().enumerate() {
                if path.contains(&i) {
                    continue;
                }
                let Some(out_set) = desc.step(caps) else { continue };
                for out in out_set.alternatives() {
                    let mut new_path = path.clone();
                    new_path.push(i);
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

    use crate::element::DynAsyncElement;
    use crate::pad_template::PadTemplates;

    /// A registered element type: its autoplug metadata plus a parameterless
    /// constructor producing a boxed transform/sink for the graph runner. The
    /// constructor is a plain `fn` pointer, the common case being a closure
    /// `|| Box::new(MyTransform::new())` coerced at the call site.
    pub struct ElementFactory {
        desc: ElementDesc,
        build: fn() -> Box<dyn DynAsyncElement>,
    }

    impl ElementFactory {
        /// Register an element type by name, pad templates, and constructor.
        pub fn new(
            name: &'static str,
            templates: Vec<PadTemplate>,
            build: fn() -> Box<dyn DynAsyncElement>,
        ) -> Self {
            Self { desc: ElementDesc::new(name, templates), build }
        }

        /// Build from a [`PadTemplates`] type, pulling its templates from the
        /// trait so the registration site names only the type and constructor.
        pub fn of<E: PadTemplates>(
            name: &'static str,
            build: fn() -> Box<dyn DynAsyncElement>,
        ) -> Self {
            Self::new(name, E::pad_templates(), build)
        }

        /// Instantiate a fresh boxed element.
        pub fn build(&self) -> Box<dyn DynAsyncElement> {
            (self.build)()
        }

        /// This factory's autoplug descriptor.
        pub fn desc(&self) -> &ElementDesc {
            &self.desc
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
    }

    impl Registry {
        /// An empty registry.
        pub fn new() -> Self {
            Self::default()
        }

        /// Register one element factory, returning `&mut self` to chain calls.
        pub fn register(&mut self, factory: ElementFactory) -> &mut Self {
            self.factories.push(factory);
            self
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
            Some(chain.into_iter().map(|i| self.factories[i].desc.name).collect())
        }

        /// Find the shortest chain converting `input` into caps satisfying
        /// `target` and instantiate it: an ordered list of boxed elements
        /// (upstream first) ready to splice onto [`run_graph`] as transforms.
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
            Some(chain.into_iter().map(|i| self.factories[i].build()).collect())
        }
    }
}

#[cfg(feature = "std")]
pub use factory::{ElementFactory, Registry};

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

    #[test]
    fn finds_single_decoder_for_h264_to_raw() {
        let descs = [parser(), decoder()];
        let chain = find_chain(&descs, &h264(Dim::Fixed(1280)), &is_raw_video, 4)
            .expect("decoder bridges H.264 to raw");
        // Shortest path is the decoder alone (the parser is same-shape, so it
        // never shortens the route to raw).
        assert_eq!(chain, Vec::from([1usize]));
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
        assert_eq!(chain, Vec::from([1usize, 2usize]), "decoder then converter");
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
}
