//! Graphviz DOT rendering for a pipeline graph (developer tooling): the
//! `GST_DEBUG_DUMP_DOT_DIR` analog. [`ValidatedGraph::to_dot`] turns a graph,
//! plus the solver's per-edge negotiated [`Caps`] and (when known) each link's
//! memory domain, into a `digraph { .. }` a developer can render with
//! `dot -Tsvg`. Self-contained and `no_std + alloc`: it only formats `String`s,
//! no I/O, so it builds on every target the core does.
//!
//! The graph carries an opaque element payload `E`, so node display names come
//! from a caller-supplied closure (the runner has each element's instance
//! name; a structural tee/muxer falls back to its kind). Edges are annotated
//! from [`DotAnnotations`], indexed by edge id to match [`ValidatedGraph::edge`]
//! and the `Vec<Caps>` that
//! [`solve_graph`](crate::runtime::solver::solve_graph) returns.

use alloc::format;
use alloc::string::{String, ToString};
use core::fmt::Write as _;

use crate::caps::Caps;
use crate::graph::{Edge, Graph, NodeId, NodeKind, ValidatedGraph};
use crate::link::LinkPolicy;
use crate::memory::MemoryDomainKind;

/// Per-edge annotations layered onto a [`ValidatedGraph`] DOT dump: the
/// solver's negotiated caps and, when the runner knows it, the link's memory
/// domain. Both are indexed by edge id (the index into
/// [`ValidatedGraph::edge`], which is also how
/// [`solve_graph`](crate::runtime::solver::solve_graph) returns its solution),
/// and either may be omitted (a pre-negotiation dump has neither). An entry is
/// rendered only when the slice covers that edge id.
#[derive(Debug, Default, Clone, Copy)]
pub struct DotAnnotations<'a> {
    /// Negotiated caps per edge, e.g. from `solve_graph`. Rendered as the
    /// edge's primary label via [`Caps::to_gst_string`].
    pub edge_caps: Option<&'a [Caps]>,
    /// Memory domain per edge. Memory domains are not part of [`Caps`] (they
    /// ride the auto-plug metadata, see DESIGN.md 4.13.9), so they are passed
    /// alongside. A non-`System` domain marks a zero-copy GPU link and is drawn
    /// bold.
    pub edge_memory: Option<&'a [MemoryDomainKind]>,
}

impl<E> ValidatedGraph<E> {
    /// Render this graph as Graphviz DOT. `label(node)` supplies a node's
    /// display name (typically the element instance name); returning `None`
    /// falls back to the node's structural kind, which is the right answer for
    /// a `tee` / `mux` that carries no element. `ann` adds negotiated caps and
    /// memory domains per edge; pass `&DotAnnotations::default()` for a bare
    /// topology dump.
    ///
    /// `title` names the `digraph` (Graphviz requires a valid identifier or
    /// quoted string; it is quoted and escaped here).
    pub fn to_dot(
        &self,
        title: &str,
        label: impl Fn(NodeId) -> Option<String>,
        ann: &DotAnnotations<'_>,
    ) -> String {
        render(
            title,
            self.node_count(),
            |n| self.kind(n),
            self.edges(),
            label,
            ann,
        )
    }
}

impl<E> Graph<E> {
    /// Render the (not-yet-validated) graph as Graphviz DOT, for a dump before
    /// `finish()` runs (a parsed launch line, a half-built graph). Same shape as
    /// [`ValidatedGraph::to_dot`]; only [`DotAnnotations`] caps/memory are
    /// usually absent pre-negotiation. Node ids `0..node_count` always exist, so
    /// the kind lookup never misses.
    pub fn to_dot(
        &self,
        title: &str,
        label: impl Fn(NodeId) -> Option<String>,
        ann: &DotAnnotations<'_>,
    ) -> String {
        render(
            title,
            self.node_count(),
            |n| self.node_kind(n).expect("node id in range"),
            self.edges(),
            label,
            ann,
        )
    }
}

/// Shared DOT body for [`Graph`] and [`ValidatedGraph`]: both supply a node
/// count, a kind lookup, and the edge slice (edge id = index, the key `ann`
/// uses).
fn render(
    title: &str,
    node_count: usize,
    kind_of: impl Fn(NodeId) -> NodeKind,
    edges: &[Edge],
    label: impl Fn(NodeId) -> Option<String>,
    ann: &DotAnnotations<'_>,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "digraph \"{}\" {{", escape(title));
    s.push_str("  rankdir=LR;\n");
    s.push_str("  node [fontname=\"monospace\", fontsize=10];\n");
    s.push_str("  edge [fontname=\"monospace\", fontsize=9];\n");

    // Nodes, in id order so the output is stable across runs.
    for i in 0..node_count {
        let node = NodeId(i as u32);
        let kind = kind_of(node);
        let name = label(node).unwrap_or_else(|| kind_label(kind).to_string());
        let _ = writeln!(
            s,
            "  n{i} [label=\"{}\"{}];",
            escape(&name),
            node_style(kind)
        );
    }

    s.push('\n');

    // Edges, in edge-id order (the index `ann` is keyed by).
    for (id, e) in edges.iter().enumerate() {
        let (src, dst) = (e.src.node.0, e.dst.node.0);
        let domain = ann.edge_memory.and_then(|m| m.get(id).copied());
        let label = edge_label(ann.edge_caps.and_then(|c| c.get(id)), domain, e.policy);
        let mut attrs = String::new();
        if !label.is_empty() {
            let _ = write!(attrs, "label=\"{}\"", escape(&label));
        }
        // A non-System domain is a GPU / zero-copy link: draw it bold so a
        // PCIe download (a System link between two GPU stages) stands out.
        if matches!(domain, Some(d) if d != MemoryDomainKind::System) {
            if !attrs.is_empty() {
                attrs.push_str(", ");
            }
            attrs.push_str("color=\"#b58900\", penwidth=2");
        }
        // Pad indices for the fan-out / fan-in nodes, so a tee's branch or a
        // muxer's input pad is identifiable.
        let pads = pad_labels(e.src.index, e.dst.index);
        if !pads.is_empty() {
            if !attrs.is_empty() {
                attrs.push_str(", ");
            }
            attrs.push_str(&pads);
        }
        if attrs.is_empty() {
            let _ = writeln!(s, "  n{src} -> n{dst};");
        } else {
            let _ = writeln!(s, "  n{src} -> n{dst} [{attrs}];");
        }
    }

    s.push_str("}\n");
    s
}

/// Default node label when the caller has no name: the structural kind.
fn kind_label(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Source => "source",
        NodeKind::Transform => "transform",
        NodeKind::Sink => "sink",
        NodeKind::Tee(_) => "tee",
        NodeKind::Muxer(_) => "mux",
        NodeKind::FaninSink(_) => "fanin-sink",
    }
}

/// Per-kind shape + fill, so the role reads at a glance (green sources, red
/// sinks, blue transforms, tan fan-out/in). Returned as the trailing
/// `, shape=.., style=..` of a node statement.
fn node_style(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Source => ", shape=box, style=\"rounded,filled\", fillcolor=\"#cde8cd\"",
        NodeKind::Sink => ", shape=box, style=\"rounded,filled\", fillcolor=\"#f0cdcd\"",
        NodeKind::Transform => ", shape=box, style=\"rounded,filled\", fillcolor=\"#cddcf0\"",
        NodeKind::Tee(_) => ", shape=diamond, style=filled, fillcolor=\"#f0e8cd\"",
        NodeKind::Muxer(_) => ", shape=trapezium, style=filled, fillcolor=\"#f0e8cd\"",
        NodeKind::FaninSink(_) => ", shape=trapezium, style=filled, fillcolor=\"#f0cdcd\"",
    }
}

/// Build an edge's label from its (optional) caps, memory domain, and policy,
/// one fact per line. Empty when nothing is known and the policy is the
/// default `Block` (a bare arrow).
fn edge_label(caps: Option<&Caps>, domain: Option<MemoryDomainKind>, policy: LinkPolicy) -> String {
    let mut lines: alloc::vec::Vec<String> = alloc::vec::Vec::new();
    if let Some(c) = caps {
        lines.push(c.to_gst_string());
    }
    if let Some(d) = domain {
        if d != MemoryDomainKind::System {
            lines.push(format!("memory:{d:?}"));
        }
    }
    if policy != LinkPolicy::Block {
        lines.push(format!("[{policy:?}]"));
    }
    // Graphviz line break inside a quoted label is the two-char sequence \n.
    lines.join("\\n")
}

/// `taillabel` / `headlabel` for a fan-out / fan-in edge, naming the pad index
/// at each end when it is not the default 0. Empty for plain 1:1 links.
fn pad_labels(src_index: u8, dst_index: u8) -> String {
    let mut parts: alloc::vec::Vec<String> = alloc::vec::Vec::new();
    if src_index != 0 {
        parts.push(format!("taillabel=\"{src_index}\""));
    }
    if dst_index != 0 {
        parts.push(format!("headlabel=\"{dst_index}\""));
    }
    parts.join(", ")
}

/// Escape a string for a quoted Graphviz attribute: backslash and double quote.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Caps, Dim, Rate, VideoCodec};
    use crate::graph::Graph;
    use crate::link::LinkPolicy;

    type G = Graph<&'static str>;

    fn h264(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    #[test]
    fn linear_chain_renders_nodes_and_caps_labelled_edges() {
        let mut g = G::new();
        let src = g.add_source("rtspsrc");
        let tx = g.add_transform("h264parse");
        let sink = g.add_sink("fakesink");
        g.link(src, tx).unwrap();
        g.link(tx, sink).unwrap();
        let v = g.finish().unwrap();

        let caps = [h264(1920, 1080), h264(1920, 1080)];
        let dot = v.to_dot(
            "pipeline",
            |n| v.element(n).map(|e| (*e).to_string()),
            &DotAnnotations {
                edge_caps: Some(&caps),
                edge_memory: None,
            },
        );

        assert!(dot.starts_with("digraph \"pipeline\" {"));
        assert!(dot.trim_end().ends_with('}'));
        // Each element name appears as a node label.
        assert!(dot.contains("label=\"rtspsrc\""));
        assert!(dot.contains("label=\"h264parse\""));
        assert!(dot.contains("label=\"fakesink\""));
        // Both edges exist and carry the negotiated caps.
        assert!(dot.contains("n0 -> n1"));
        assert!(dot.contains("n1 -> n2"));
        assert!(
            dot.contains("video/x-h264"),
            "edge caps should be labelled: {dot}"
        );
        // Source/sink get distinct fills.
        assert!(dot.contains("fillcolor=\"#cde8cd\"")); // source green
        assert!(dot.contains("fillcolor=\"#f0cdcd\"")); // sink red
    }

    #[test]
    fn structural_nodes_fall_back_to_kind_and_pads_are_labelled() {
        let mut g = G::new();
        let src = g.add_source("src");
        let tee = g.add_tee(2);
        let a = g.add_sink("a");
        let b = g.add_sink("b");
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), a).unwrap();
        // Branch 1 is a leaky preview branch.
        g.link_with(tee.out(1), b, LinkPolicy::DropOldest).unwrap();
        let v = g.finish().unwrap();

        // The closure has no name for the tee node, so it falls back to "tee".
        let dot = v.to_dot(
            "fanout",
            |n| v.element(n).map(|e| (*e).to_string()),
            &DotAnnotations::default(),
        );
        assert!(
            dot.contains("label=\"tee\""),
            "tee uses kind fallback: {dot}"
        );
        assert!(dot.contains("shape=diamond"));
        // The second tee output pad (index 1) is named, and its leaky policy shows.
        assert!(
            dot.contains("taillabel=\"1\""),
            "tee branch pad index: {dot}"
        );
        assert!(
            dot.contains("[DropOldest]"),
            "non-default policy shown: {dot}"
        );
    }

    #[test]
    fn gpu_memory_edge_is_marked() {
        let mut g = G::new();
        let src = g.add_source("nvdec");
        let sink = g.add_sink("nvenc");
        g.link(src, sink).unwrap();
        let v = g.finish().unwrap();

        let mem = [MemoryDomainKind::Cuda];
        let dot = v.to_dot(
            "gpu",
            |n| v.element(n).map(|e| (*e).to_string()),
            &DotAnnotations {
                edge_caps: None,
                edge_memory: Some(&mem),
            },
        );
        assert!(dot.contains("memory:Cuda"), "CUDA domain labelled: {dot}");
        assert!(dot.contains("penwidth=2"), "GPU link drawn bold: {dot}");
        // A System domain would not be annotated.
        let sys = [MemoryDomainKind::System];
        let dot2 = v.to_dot(
            "sys",
            |_| None,
            &DotAnnotations {
                edge_caps: None,
                edge_memory: Some(&sys),
            },
        );
        assert!(
            !dot2.contains("memory:"),
            "System domain is not labelled: {dot2}"
        );
    }

    #[test]
    fn title_and_names_are_escaped() {
        let mut g = G::new();
        let src = g.add_source("a\"b");
        let sink = g.add_sink("sink");
        g.link(src, sink).unwrap();
        let v = g.finish().unwrap();
        let dot = v.to_dot(
            "t\"t",
            |n| v.element(n).map(|e| (*e).to_string()),
            &DotAnnotations::default(),
        );
        assert!(
            dot.contains("digraph \"t\\\"t\""),
            "title quote escaped: {dot}"
        );
        assert!(
            dot.contains("label=\"a\\\"b\""),
            "name quote escaped: {dot}"
        );
    }
}
