//! Copy / allocation plan (M613): a static, pre-run analysis of the memory-domain
//! path a frame takes through a negotiated graph.
//!
//! g2g's whole-graph negotiation already resolves, per edge, the memory domain the
//! frame lives in (the producer's `output_memory`, see
//! `runtime::graph_runner::negotiate_graph`). This module turns that implicit
//! knowledge into an explicit, checkable artifact: the sequence of memory *hops* and
//! the *transfers* between them, so a pipeline can prove a property GStreamer cannot
//! state at construction time:
//!
//! > "This graph keeps every frame on the GPU end to end: zero host round-trips."
//!
//! A **transfer** is a node whose output domain differs from the domain it consumed
//! (the framework's domain converters live here: a CUDA upload, an NVDEC download, a
//! `WgpuToDmaBuf` export). Each transfer is classified by cost
//! ([`TransferKind`]) and flagged as a real **frame copy** only when a raw heavy
//! buffer (raw video / PCM audio / tensor, see [`Caps::is_raw_media`]) crosses the
//! boundary on both sides. A decode (`CompressedVideo` -> `RawVideo`) or encode
//! (`RawVideo` -> `CompressedVideo`) changes domain without copying a raw frame, so
//! it is surfaced in the trace but not counted as a copy.
//!
//! [`CopyPlan::check`] enforces a [`CopyPolicy`] as a graph-level contract: a graph
//! that exceeds its copy budget fails the check, so an accidental host round-trip in
//! a zero-copy pipeline is caught before the pipeline runs, not measured after.
//!
//! The analysis is pure (no graph or element types): it works over the flat
//! [`NodeProfile`] / [`EdgeProfile`] arrays the runner extracts from a negotiated
//! graph, mirroring how [`crate::dot`] takes flat annotations.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::caps::Caps;
use crate::memory::MemoryDomainKind;

/// One node's memory profile for the copy analysis.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    /// Display label (the element's log category, or the structural kind).
    pub label: String,
    /// The memory domain the node emits on its output (`System` if it has none).
    pub out_domain: MemoryDomainKind,
}

/// One negotiated edge: the frame's domain and fixated caps as it leaves the
/// producer (`src`) for the consumer (`dst`), indexed into the node array.
#[derive(Debug, Clone)]
pub struct EdgeProfile {
    /// Producer node index.
    pub src: usize,
    /// Consumer node index.
    pub dst: usize,
    /// The memory domain the frame occupies on this edge (the producer's output).
    pub domain: MemoryDomainKind,
    /// The fixated caps on this edge.
    pub caps: Caps,
}

/// The cost class of moving a frame from one memory domain to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferKind {
    /// Same domain, or both host-side: the frame is handed over by reference / a
    /// negligible host operation. No device transfer.
    None,
    /// Crosses a boundary designed for zero-copy sharing (a dma-buf import/export,
    /// or a device-to-device interop bridge). No guaranteed byte copy, but the
    /// boundary is flagged so a strict budget can still see it.
    Interop,
    /// A device-to-host transfer over the system bus (a GPU download or upload of
    /// the frame bytes). The expensive copy, and the one a zero-copy GPU pipeline
    /// exists to avoid.
    DeviceHost,
    /// A copy between two distinct device domains with no known zero-copy path.
    CrossDevice,
}

impl TransferKind {
    /// Whether this transfer moves bytes (device-to-host or cross-device), as
    /// opposed to a free handoff ([`None`](Self::None)) or a zero-copy
    /// [`Interop`](Self::Interop) share.
    pub fn copies_bytes(self) -> bool {
        matches!(self, TransferKind::DeviceHost | TransferKind::CrossDevice)
    }
}

/// Whether a domain is host (CPU) memory.
fn is_host(k: MemoryDomainKind) -> bool {
    matches!(k, MemoryDomainKind::System | MemoryDomainKind::SystemView)
}

/// Classify the cost of moving a frame from domain `from` to domain `to`.
pub fn classify(from: MemoryDomainKind, to: MemoryDomainKind) -> TransferKind {
    if from == to {
        return TransferKind::None;
    }
    // dma-buf exists precisely for zero-copy sharing (across the CPU/GPU line or
    // between devices), so any hop involving it is interop, not a byte copy.
    if from == MemoryDomainKind::DmaBuf || to == MemoryDomainKind::DmaBuf {
        return TransferKind::Interop;
    }
    match (is_host(from), is_host(to)) {
        // Both host (System <-> SystemView): a view or a cheap CPU touch, no
        // device bus transfer.
        (true, true) => TransferKind::None,
        // Exactly one side is host: a GPU download or upload over the bus.
        (true, false) | (false, true) => TransferKind::DeviceHost,
        // Two distinct device domains with no dma-buf bridge: a staging copy
        // (may be zero-copy interop depending on the bridge element; flagged).
        (false, false) => TransferKind::CrossDevice,
    }
}

/// A memory-domain transition at a node: it consumed `from` and emits `to`.
#[derive(Debug, Clone)]
pub struct Transfer {
    /// Node index where the domain changes.
    pub at: usize,
    /// The node's display label.
    pub label: String,
    /// The domain consumed on the input edge.
    pub from: MemoryDomainKind,
    /// The domain emitted on the output edge.
    pub to: MemoryDomainKind,
    /// The transfer's cost class.
    pub kind: TransferKind,
    /// Whether a raw heavy buffer (raw video / PCM audio / tensor) crosses the
    /// boundary on both sides: a real frame copy, as opposed to a codec boundary.
    pub frame_copy: bool,
}

impl Transfer {
    /// Whether this transfer is a real, counted frame copy: a byte-copying
    /// transfer of a raw heavy buffer (not a free handoff, zero-copy interop, or a
    /// codec boundary).
    pub fn is_counted_copy(&self) -> bool {
        self.frame_copy && self.kind.copies_bytes()
    }
}

/// One memory hop: the domain a frame occupies on one edge, producer to consumer.
#[derive(Debug, Clone)]
pub struct Hop {
    /// Producer label.
    pub src_label: String,
    /// Consumer label.
    pub dst_label: String,
    /// The domain the frame lives in on this hop.
    pub domain: MemoryDomainKind,
}

/// The memory-domain path through a negotiated graph: the per-edge [`Hop`]s and the
/// [`Transfer`]s between differing domains.
#[derive(Debug, Clone)]
pub struct CopyPlan {
    /// Per-edge memory hops, in edge order.
    pub hops: Vec<Hop>,
    /// Domain transitions (the converter / transfer points), in node order.
    pub transfers: Vec<Transfer>,
}

/// A graph-level budget on memory-domain copies, enforced by [`CopyPlan::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyPolicy {
    /// Report only; the check always passes.
    Allow,
    /// Pass only if the plan has at most `n` counted frame copies.
    AtMost(u8),
    /// Pass only with zero frame copies: the strict zero-copy contract
    /// (equivalent to `AtMost(0)`).
    DenyAll,
}

/// A [`CopyPolicy`] violation: the plan had more frame copies than the budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyBudgetError {
    /// Counted frame copies found in the plan.
    pub copies: usize,
    /// The budget the policy allowed.
    pub budget: usize,
    /// A short description of each counted copy (`"node: From -> To"`).
    pub offenders: Vec<String>,
}

impl core::fmt::Display for CopyBudgetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "copy budget exceeded: {} frame copies, budget {} ({})",
            self.copies,
            self.budget,
            self.offenders.join(", ")
        )
    }
}

impl CopyPlan {
    /// Analyze a negotiated graph's flat node / edge profiles into a copy plan.
    ///
    /// A transfer is recorded at every node whose output domain differs from the
    /// domain it consumed on an input edge; it is a counted frame copy when that
    /// input edge and the node's output both carry raw media
    /// ([`Caps::is_raw_media`]) and the transfer copies bytes.
    pub fn analyze(nodes: &[NodeProfile], edges: &[EdgeProfile]) -> CopyPlan {
        let label = |i: usize| nodes.get(i).map(|n| n.label.clone()).unwrap_or_default();

        let hops = edges
            .iter()
            .map(|e| Hop {
                src_label: label(e.src),
                dst_label: label(e.dst),
                domain: e.domain,
            })
            .collect();

        let mut transfers = Vec::new();
        for (idx, node) in nodes.iter().enumerate() {
            let out = node.out_domain;
            // The caps this node emits (from any of its output edges), used to tell
            // a raw-frame transfer from a codec boundary.
            let out_caps = edges.iter().find(|e| e.src == idx).map(|e| &e.caps);
            // Each input edge is a potential domain transition into this node.
            for e in edges.iter().filter(|e| e.dst == idx) {
                let kind = classify(e.domain, out);
                if kind == TransferKind::None {
                    continue;
                }
                let frame_copy =
                    e.caps.is_raw_media() && out_caps.is_some_and(|c| c.is_raw_media());
                transfers.push(Transfer {
                    at: idx,
                    label: node.label.clone(),
                    from: e.domain,
                    to: out,
                    kind,
                    frame_copy,
                });
            }
        }
        CopyPlan { hops, transfers }
    }

    /// The number of counted frame copies: byte-copying transfers of a raw heavy
    /// buffer. This is what [`CopyPolicy`] budgets against.
    pub fn frame_copies(&self) -> usize {
        self.transfers.iter().filter(|t| t.is_counted_copy()).count()
    }

    /// The number of device-to-host round trips of a raw frame (the PCIe copies):
    /// a subset of [`frame_copies`](Self::frame_copies).
    pub fn host_round_trips(&self) -> usize {
        self.transfers
            .iter()
            .filter(|t| t.frame_copy && t.kind == TransferKind::DeviceHost)
            .count()
    }

    /// Whether the graph is zero-copy: no counted frame copies.
    pub fn is_zero_copy(&self) -> bool {
        self.frame_copies() == 0
    }

    /// Enforce a [`CopyPolicy`] as a graph-level contract.
    pub fn check(&self, policy: CopyPolicy) -> Result<(), CopyBudgetError> {
        let budget = match policy {
            CopyPolicy::Allow => return Ok(()),
            CopyPolicy::DenyAll => 0,
            CopyPolicy::AtMost(n) => n as usize,
        };
        let copies = self.frame_copies();
        if copies <= budget {
            return Ok(());
        }
        let offenders = self
            .transfers
            .iter()
            .filter(|t| t.is_counted_copy())
            .map(|t| format!("{}: {:?} -> {:?}", t.label, t.from, t.to))
            .collect();
        Err(CopyBudgetError { copies, budget, offenders })
    }

    /// A human-readable report: the per-hop domain trace with each transfer marked
    /// (`!` for a counted frame copy, `~` for a zero-copy interop / codec
    /// boundary), and a one-line verdict.
    pub fn to_report(&self) -> String {
        let mut s = String::new();
        let copies = self.frame_copies();
        let verdict = if copies == 0 {
            "zero-copy".to_string()
        } else {
            format!("{copies} frame cop{}", if copies == 1 { "y" } else { "ies" })
        };
        s.push_str(&format!("Copy plan ({verdict}):\n"));
        for hop in &self.hops {
            s.push_str(&format!(
                "  {} --{:?}--> {}\n",
                hop.src_label, hop.domain, hop.dst_label
            ));
        }
        if !self.transfers.is_empty() {
            s.push_str("  transfers:\n");
            for t in &self.transfers {
                let mark = if t.is_counted_copy() { "!" } else { "~" };
                let note = match (t.kind, t.frame_copy) {
                    (TransferKind::DeviceHost, true) => "device<->host copy (raw frame)",
                    (TransferKind::CrossDevice, true) => "cross-device copy (raw frame)",
                    (TransferKind::DeviceHost, false) | (TransferKind::CrossDevice, false) => {
                        "codec boundary (no raw-frame copy)"
                    }
                    (TransferKind::Interop, _) => "zero-copy interop",
                    (TransferKind::None, _) => "",
                };
                s.push_str(&format!(
                    "    {mark} {}  {:?} -> {:?}  {note}\n",
                    t.label, t.from, t.to
                ));
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Caps, RawVideoFormat, VideoCodec};
    use crate::memory::MemoryDomainKind::*;
    use crate::{Dim, Rate};

    fn raw(domain: MemoryDomainKind) -> EdgeProfile {
        EdgeProfile {
            src: 0,
            dst: 0,
            domain,
            caps: Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(64),
                height: Dim::Fixed(64),
                framerate: Rate::Fixed(30 << 16),
            },
        }
    }

    fn compressed(domain: MemoryDomainKind) -> EdgeProfile {
        EdgeProfile {
            src: 0,
            dst: 0,
            domain,
            caps: Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(64),
                height: Dim::Fixed(64),
                framerate: Rate::Fixed(30 << 16),
            },
        }
    }

    fn node(label: &str, out: MemoryDomainKind) -> NodeProfile {
        NodeProfile { label: label.to_string(), out_domain: out }
    }

    /// Wire up a linear chain: nodes[i] -> nodes[i+1], each edge carrying the given
    /// (caps-template, domain) via the provided EdgeProfile whose src/dst we fix.
    fn chain(nodes: &[NodeProfile], mut edges: Vec<EdgeProfile>) -> CopyPlan {
        for (i, e) in edges.iter_mut().enumerate() {
            e.src = i;
            e.dst = i + 1;
        }
        CopyPlan::analyze(nodes, &edges)
    }

    #[test]
    fn classify_covers_the_domain_families() {
        assert_eq!(classify(System, System), TransferKind::None);
        assert_eq!(classify(System, SystemView), TransferKind::None, "host<->host is free");
        assert_eq!(classify(System, Cuda), TransferKind::DeviceHost, "upload");
        assert_eq!(classify(Cuda, System), TransferKind::DeviceHost, "download");
        assert_eq!(classify(Cuda, WgpuTexture), TransferKind::CrossDevice);
        assert_eq!(classify(Cuda, DmaBuf), TransferKind::Interop, "dma-buf is zero-copy");
        assert_eq!(classify(DmaBuf, System), TransferKind::Interop);
    }

    #[test]
    fn all_system_pipeline_is_zero_copy() {
        // filesrc -> parse -> dec -> sink, all in System memory: no transfers.
        let nodes = [node("filesrc", System), node("h264parse", System), node("dec", System), node("sink", System)];
        let plan = chain(&nodes, alloc::vec![compressed(System), compressed(System), raw(System)]);
        assert!(plan.is_zero_copy());
        assert_eq!(plan.frame_copies(), 0);
        assert!(plan.transfers.is_empty(), "no domain changes");
        assert!(plan.check(CopyPolicy::DenyAll).is_ok());
    }

    #[test]
    fn gpu_resident_pipeline_stays_zero_copy_across_a_decode() {
        // nvdec decodes compressed(System) -> raw(Cuda), then cudascale stays on
        // Cuda, then a cuda sink. The decode changes domain (System->Cuda) but the
        // input is compressed, so it is a codec boundary, not a counted frame copy.
        let nodes = [
            node("filesrc", System),
            node("nvh264dec", Cuda),
            node("cudascale", Cuda),
            node("cudasink", Cuda),
        ];
        let plan = chain(&nodes, alloc::vec![compressed(System), raw(Cuda), raw(Cuda)]);
        assert_eq!(plan.frame_copies(), 0, "decode-into-device is not a raw-frame copy");
        assert!(plan.is_zero_copy());
        // The transition is still surfaced in the trace.
        assert_eq!(plan.transfers.len(), 1);
        assert_eq!(plan.transfers[0].kind, TransferKind::DeviceHost);
        assert!(!plan.transfers[0].frame_copy);
        assert!(plan.check(CopyPolicy::DenyAll).is_ok());
    }

    #[test]
    fn a_host_download_of_a_raw_frame_is_a_counted_copy() {
        // A GPU decoder that downloads: raw(Cuda) -> a videoconvert that emits
        // raw(System). Both sides raw, device->host: a real PCIe copy.
        let nodes = [
            node("nvdec", Cuda),
            node("download", System), // consumes raw Cuda, emits raw System
            node("filesink", System),
        ];
        let plan = chain(&nodes, alloc::vec![raw(Cuda), raw(System)]);
        assert_eq!(plan.frame_copies(), 1);
        assert_eq!(plan.host_round_trips(), 1);
        assert!(!plan.is_zero_copy());
        let err = plan.check(CopyPolicy::DenyAll).unwrap_err();
        assert_eq!(err.copies, 1);
        assert_eq!(err.budget, 0);
        assert_eq!(err.offenders.len(), 1);
        assert!(err.offenders[0].contains("download"));
        // A budget of one copy tolerates it.
        assert!(plan.check(CopyPolicy::AtMost(1)).is_ok());
    }

    #[test]
    fn encode_off_gpu_is_not_a_frame_copy() {
        // nvenc reads raw(Cuda) and emits compressed(System): the raw frame is
        // consumed on-device, only the small bitstream lands in System. Domain
        // changes, but the output is not raw -> not a counted copy.
        let nodes = [node("cudasrc", Cuda), node("nvenc", System), node("filesink", System)];
        let plan = chain(&nodes, alloc::vec![raw(Cuda), compressed(System)]);
        assert_eq!(plan.frame_copies(), 0);
        assert!(plan.is_zero_copy());
    }

    #[test]
    fn report_marks_the_offending_copy() {
        let nodes = [node("nvdec", Cuda), node("download", System), node("sink", System)];
        let plan = chain(&nodes, alloc::vec![raw(Cuda), raw(System)]);
        let report = plan.to_report();
        assert!(report.contains("1 frame copy"), "verdict counts the copy:\n{report}");
        assert!(report.contains("! download"), "the copy is flagged:\n{report}");
        assert!(report.contains("Cuda"), "trace shows the GPU hop:\n{report}");
    }
}
