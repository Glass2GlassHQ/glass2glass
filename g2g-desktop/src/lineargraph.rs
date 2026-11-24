// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::runtime::GraphNode;
use g2g_core::Graph;

pub(crate) fn linear_graph(
    source: GraphNode,
    transforms: impl IntoIterator<Item = GraphNode>,
    sink: GraphNode,
) -> Graph<GraphNode> {
    let mut graph = Graph::new();
    let mut previous = graph.add_source(source);
    for transform in transforms {
        let node = graph.add_transform(transform);
        graph
            .link(previous, node)
            .expect("transforms have one input and one output pad");
        previous = node;
    }
    let sink = graph.add_sink(sink);
    graph
        .link(previous, sink)
        .expect("sinks have one input pad");
    graph
}
