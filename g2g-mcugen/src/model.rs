//! The MCU graph document schema. Deliberately a *different* schema from the
//! dynamic `g2g_plugins::declarative::GraphSpec`, not a copy: this one carries
//! the frame geometry a static build needs (`frame_ns`, `frames`, per-source
//! sample format) and omits the dynamic-only escape hatches that cannot
//! monomorphize (a `pipeline:` launch string, tee/demux fan-out, per-edge
//! backpressure policies). The overlap, nodes with an `id` / `element` /
//! `props` plus `from` -> `to` edges, is the universal shape of a graph
//! document, not shared machinery.

use std::collections::BTreeMap;

use serde::Deserialize;

/// A format-agnostic scalar property value (deserializes the same from JSON
/// and YAML). The element catalog interprets each against the property it
/// names (a rate as `u32`, a gain as `i16`, a law as a string).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Str(String),
}

impl Scalar {
    /// The value as an integer, if it is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Scalar::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// The value as a string slice, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Scalar::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One graph node: a unique `id`, the catalog `element` kind, and its typed
/// `props`.
#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub id: String,
    pub element: String,
    #[serde(default)]
    pub props: BTreeMap<String, Scalar>,
}

/// One directed edge from `from`'s output to `to`'s input.
#[derive(Debug, Clone, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
}

/// A whole graph document.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphDoc {
    /// Identifier used to name the generated module surface (snake_case
    /// recommended; validated to be a Rust-identifier-safe token).
    pub name: String,
    /// Frame period in nanoseconds (the capture cadence; drives PTS and the
    /// per-node sample counts).
    pub frame_ns: u64,
    /// Number of frames each source emits before end of stream.
    pub frames: u32,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
