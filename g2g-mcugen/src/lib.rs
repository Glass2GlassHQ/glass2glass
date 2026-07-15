//! Host graph compiler (M646, generalized M648): turn a declarative MCU graph
//! document into a monomorphized static pipeline, the codegen resolution of the
//! "develop-and-test on Linux, ship a bounded static build to the MCU" tension.
//! The same elements the hand-written flagship pipelines wire by hand, this
//! compiler wires from a `graph.yaml`, computing each ring's size from the
//! graph's frame geometry instead of hard-coding it, and reporting the total
//! static ring budget.
//!
//! The compiler is not audio-specific: its catalog spans an audio chain
//! (`capture -> convert -> resample -> mix -> encode -> RTP`, the hand-written
//! `noalloc-pipeline::audio`) and a video / display chain (`camera -> SPI
//! display`, `noalloc-pipeline`'s reference pipeline). Frame geometry is a sum
//! of audio (rate / width / channels) and raster (pixels / bpp), and the sink
//! seam varies per sink kind (an RTP packet sender, or an SPI bus + D/C pin +
//! delay), so a board or proof harness supplies the right HAL impls.
//!
//! The generated code is ordinary `no_std` Rust using the `g2g_core`
//! static-element runners and `g2g_mcu` elements: it carries the same
//! heap-free / panic-free / footprint guarantees as the references, which the
//! `examples/mcugen-graphs` proof confirms by regenerating both flagship graphs
//! and checking each one's wire output byte-for-byte against its reference
//! checksum (`AUDIO_EXPECTED_CHECKSUM` / `EXPECTED_CHECKSUM`).
//!
//! This is a host developer tool. It emits source; it never runs on the MCU.

mod catalog;
mod codegen;
mod model;

pub use codegen::Compiled;
pub use model::GraphDoc;

use std::fmt;

/// Why a document could not be compiled.
#[derive(Debug)]
pub enum CompileError {
    /// The document did not parse as YAML/JSON into the schema.
    Parse(String),
    /// `name` is not a safe identifier token.
    BadName(String),
    /// The document has no nodes.
    Empty,
    /// Two nodes share an `id`.
    DuplicateId(String),
    /// An edge names a node that does not exist.
    UnknownNode(String),
    /// An element name is not in the catalog.
    UnknownElement(String),
    /// A property is missing.
    MissingProp(String),
    /// A property has the wrong type or an out-of-range value.
    BadProp { key: String, detail: String },
    /// A node needs an input link but has none.
    MissingInput(String),
    /// A geometry contract was violated (e.g. an encoder fed 32-bit slots).
    BadGeometry { node: String, detail: String },
    /// The rate and frame period do not yield a whole sample count.
    FractionalFrame { rate: u32, frame_ns: u64 },
    /// The two fan-in inputs have different geometry.
    MixerInputMismatch { a: String, b: String },
    /// The graph is not a supported topology (linear or single fan-in).
    Topology(String),
    /// A referenced node index was not resolved (internal invariant).
    Internal(&'static str),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Parse(e) => write!(f, "parse error: {e}"),
            CompileError::BadName(n) => write!(f, "graph name `{n}` is not a valid identifier"),
            CompileError::Empty => write!(f, "the graph has no nodes"),
            CompileError::DuplicateId(id) => write!(f, "duplicate node id `{id}`"),
            CompileError::UnknownNode(id) => write!(f, "edge references unknown node `{id}`"),
            CompileError::UnknownElement(e) => write!(f, "unknown element `{e}`"),
            CompileError::MissingProp(k) => write!(f, "missing required property `{k}`"),
            CompileError::BadProp { key, detail } => write!(f, "property `{key}`: {detail}"),
            CompileError::MissingInput(id) => write!(f, "node `{id}` needs an input but has none"),
            CompileError::BadGeometry { node, detail } => write!(f, "node `{node}`: {detail}"),
            CompileError::FractionalFrame { rate, frame_ns } => {
                write!(f, "rate {rate} Hz over {frame_ns} ns is not a whole number of samples")
            }
            CompileError::MixerInputMismatch { a, b } => {
                write!(f, "fan-in inputs disagree: {a} vs {b}")
            }
            CompileError::Topology(m) => write!(f, "unsupported topology: {m}"),
            CompileError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for CompileError {}

fn valid_ident(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// Parse a graph document from YAML (a superset of JSON, so both load).
pub fn parse(text: &str) -> Result<GraphDoc, CompileError> {
    let doc: GraphDoc = serde_yaml::from_str(text).map_err(|e| CompileError::Parse(e.to_string()))?;
    if !valid_ident(&doc.name) {
        return Err(CompileError::BadName(doc.name));
    }
    Ok(doc)
}

/// Compile a parsed document into generated Rust plus its ring budget.
pub fn compile(doc: &GraphDoc) -> Result<Compiled, CompileError> {
    codegen::compile(doc)
}

/// Parse and compile in one step.
pub fn compile_str(text: &str) -> Result<Compiled, CompileError> {
    compile(&parse(text)?)
}
