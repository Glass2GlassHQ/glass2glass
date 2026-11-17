// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::time::Duration;

use g2g_core::graph::{Graph, NodeId};
use g2g_core::runtime::{GraphMutator, GraphNode, GraphNodeRef};
use g2g_core::{AsyncElement, PropValue};
use g2g_plugins::level::Level;

pub(crate) const LEVEL_METER: &str = "playerlevelmeter";
const AUDIO_SINK_CLASSIFICATION: &str = "Sink/Audio";
const AUDIO_CONVERTER_CLASSIFICATION: &str = "Filter/Converter/Audio";

#[derive(Debug, Clone, PartialEq)]
pub struct AudioLevels {
    pub peak_db: Vec<f64>,
    pub rms_db: Vec<f64>,
}

fn classification(graph: &Graph<GraphNode>, node: NodeId) -> Option<&'static str> {
    match graph.element(node)? {
        GraphNodeRef::Element(element) => Some(element.metadata().classification),
        _ => None,
    }
}

fn edge_into(graph: &Graph<GraphNode>, node: NodeId) -> Option<usize> {
    graph.edges().iter().position(|edge| edge.dst.node == node)
}

fn interval_ns(window: Duration) -> u64 {
    u64::try_from(window.as_nanos()).unwrap_or(u64::MAX)
}

pub(crate) fn insert_level_meter(graph: &mut Graph<GraphNode>, window: Option<Duration>) -> bool {
    let Some(sink) = (0..graph.node_count() as u32)
        .map(NodeId)
        .find(|&node| classification(graph, node) == Some(AUDIO_SINK_CLASSIFICATION))
    else {
        return false;
    };
    let mut into = edge_into(graph, sink).expect("parse_launch links every sink");
    // playbin converts to the sink's channel count, and the meter reads the decoded channels
    loop {
        let producer = graph.edges()[into].src.node;
        if classification(graph, producer) != Some(AUDIO_CONVERTER_CLASSIFICATION) {
            break;
        }
        match edge_into(graph, producer) {
            Some(edge) => into = edge,
            None => break,
        }
    }
    let mut level = Level::new();
    level
        .set_property("post-messages", PropValue::Bool(window.is_some()))
        .expect("level takes post-messages");
    if let Some(window) = window {
        level
            .set_property("interval", PropValue::Uint(interval_ns(window)))
            .expect("level takes a non-zero interval");
    }
    let meter = graph.insert_on_edge(into, GraphNodeRef::element(level));
    graph.set_node_name(meter, LEVEL_METER.to_string());
    true
}

pub(crate) async fn apply_level_meter(mutator: &GraphMutator<'_>, window: Option<Duration>) {
    let target = g2g_core::log::Target::category("g2g-player");
    let settings = [
        ("post-messages", Some(PropValue::Bool(window.is_some()))),
        (
            "interval",
            window.map(|window| PropValue::Uint(interval_ns(window))),
        ),
    ];
    for (name, value) in settings {
        let Some(value) = value else {
            continue;
        };
        if let Err(error) = mutator.set_property(LEVEL_METER, name, value).await {
            g2g_core::g2g_error!(target, "cannot set the level meter's {name}: {error:?}");
        }
    }
}
