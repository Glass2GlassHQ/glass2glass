//! Flight recorder for pipeline errors (M1016).
//!
//! An hour into a live run something fails and the traffic that caused it is
//! gone. A [`FlightRecorder`] handed to
//! [`run_graph_recorded`](crate::runtime::run_graph_recorded) keeps a small
//! bounded ring of the most recent packets per graph edge while the run goes,
//! and [`dump_to_dir`](FlightRecorder::dump_to_dir) writes those rings out as
//! recording files once the run has failed, so the last moments of a crash
//! replay through `replaysrc` as a repro.
//!
//! Packets are serialized ([`crate::wire::encode_packet`]) at capture rather
//! than cloned: a `Frame` is deliberately not `Clone` and sharing owned CPU
//! bytes deep-copies them anyway, so encoding costs the same one copy while
//! bounding the ring in exact bytes, refusing a device-resident frame up front,
//! and leaving nothing device-owned alive in the ring after the run is torn down.
//!
//! The dump is a directory of `[u32-le length][encode_packet body]` records, one
//! file per edge, the format `replaysrc` reads. Each file leads with the
//! `CapsChanged` in effect for its oldest retained packet, so the replay is
//! typed even when the negotiated caps were refined mid-run and that refinement
//! has since scrolled out of the ring.
//!
//! Capture itself only needs `alloc`, but the module rides the `std`-gated
//! graph runner, its only caller, so the whole file is `std`-gated with it.

extern crate std;

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::{collections::VecDeque, format};

use spin::Mutex;

use crate::caps::Caps;
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::log::short_type_name;
use crate::runtime::channel::{LinkInterceptor, ProbeAction, ProbeSlot};
use crate::wire::encode_packet;

/// Packets retained per edge. At 30 fps this is the last two seconds of an edge,
/// which is the window a "what was flowing when it died" repro needs; the
/// byte bound below is what actually caps memory on a high-rate edge.
pub const FLIGHT_RING_PACKETS: usize = 60;

/// Encoded bytes retained per edge, the memory bound that holds regardless of
/// frame size: 8 MiB is ~8 seconds of a 1080p stream at 8 Mbit/s (so the packet
/// count binds first there), and two raw 1080p NV12 frames (so the byte count
/// binds first on an uncompressed edge). The newest packet is always kept even
/// if it alone exceeds this.
pub const FLIGHT_RING_BYTES: usize = 8 * 1024 * 1024;

/// File extension of a dumped edge recording, the same format `recordsink`
/// writes and `replaysrc` reads.
const DUMP_EXTENSION: &str = "g2grec";

/// Stands in for the instance name of a structural node (a plain tee) that has
/// none, so a dump file still says which hop it came from.
const UNNAMED_NODE_PREFIX: &str = "node";

/// Character substituted for anything in an element instance name that has no
/// business in a file name.
const NAME_REPLACEMENT: char = '_';

/// A bounded ring of the most recent packets on one graph edge, filled through
/// the edge's [`ProbeSlot`] (the same per-edge tap a pad probe uses).
#[derive(Debug)]
struct EdgeRing {
    state: Mutex<RingState>,
}

#[derive(Debug)]
struct RingState {
    entries: VecDeque<RingEntry>,
    bytes: usize,
    /// Encoded `CapsChanged` in effect for the oldest retained entry: the
    /// negotiated caps until a mid-run change scrolls out of the window, then
    /// that change. Leads the dump so the replay is typed.
    leading_caps: Vec<u8>,
    /// A packet on this edge could not be serialized (a device-resident frame),
    /// so the edge is not replayable and is skipped at dump time.
    unserializable: bool,
}

#[derive(Debug)]
struct RingEntry {
    bytes: Vec<u8>,
    /// This entry is a `CapsChanged`, so evicting it moves the dump's leading
    /// caps forward to it.
    caps: bool,
}

impl RingState {
    /// The dump's payloads in order: the caps in effect for the oldest retained
    /// packet, then the retained window itself.
    fn records(&self) -> impl Iterator<Item = &[u8]> {
        core::iter::once(self.leading_caps.as_slice())
            .chain(self.entries.iter().map(|e| e.bytes.as_slice()))
    }

    /// [`records`](Self::records) framed as a recording file's bytes. Built in
    /// memory so the dump holds the ring lock only for the copy, never across the
    /// file write (an arm still capturing would spin on it).
    fn framed_dump(&self) -> Result<Vec<u8>, G2gError> {
        let mut out = Vec::new();
        for payload in self.records() {
            let prefix = crate::wire::record_length_prefix(payload.len())
                .map_err(|_| G2gError::UnsupportedDomain)?;
            out.extend_from_slice(&prefix);
            out.extend_from_slice(payload);
        }
        Ok(out)
    }
}

impl EdgeRing {
    fn new(leading_caps: Vec<u8>) -> Self {
        Self {
            state: Mutex::new(RingState {
                entries: VecDeque::new(),
                bytes: 0,
                leading_caps,
                unserializable: false,
            }),
        }
    }

    fn capture(&self, packet: &PipelinePacket) {
        // Neither belongs in a recording: a Tick is a runner-internal deadline
        // with no wire form at all, and a stored Eos would end the replay's
        // consumer before the replay source pushes its own, losing whatever
        // followed. Skipped rather than failing the edge.
        if matches!(packet, PipelinePacket::Tick | PipelinePacket::Eos) {
            return;
        }
        let Ok(bytes) = encode_packet(packet) else {
            self.state.lock().unserializable = true;
            return;
        };
        let entry = RingEntry {
            caps: matches!(packet, PipelinePacket::CapsChanged(_)),
            bytes,
        };
        let mut state = self.state.lock();
        state.bytes += entry.bytes.len();
        state.entries.push_back(entry);
        while state.entries.len() > FLIGHT_RING_PACKETS
            || (state.bytes > FLIGHT_RING_BYTES && state.entries.len() > 1)
        {
            let Some(evicted) = state.entries.pop_front() else {
                break;
            };
            state.bytes -= evicted.bytes.len();
            if evicted.caps {
                state.leading_caps = evicted.bytes;
            }
        }
    }
}

impl LinkInterceptor for EdgeRing {
    fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction {
        self.capture(packet);
        ProbeAction::Pass
    }
}

/// One recorded edge: the hop it names and its ring.
#[derive(Debug)]
struct RecordedEdge {
    /// `<producer>-to-<consumer>`, from the element instance names the runner
    /// assigned.
    label: String,
    ring: Arc<EdgeRing>,
}

/// A bounded per-edge packet history, kept for the moment a run fails.
///
/// Construct one, hand it to
/// [`run_graph_recorded`](crate::runtime::run_graph_recorded) (or
/// [`run_graph_observed_recorded`](crate::runtime::run_graph_observed_recorded)),
/// and on `Err` call [`dump_to_dir`](Self::dump_to_dir) for a replayable file
/// per edge. Nothing is allocated and no packet is touched unless a run was
/// given the recorder: the rings hang off the per-edge probe slots the links
/// already carry, which stay empty (and free) otherwise.
///
/// # Example
///
/// ```no_run
/// use g2g_core::runtime::FlightRecorder;
///
/// let recorder = FlightRecorder::new();
/// assert_eq!(recorder.edge_count(), 0); // nothing until a run attaches
/// ```
#[derive(Debug, Default)]
pub struct FlightRecorder {
    edges: Mutex<Vec<RecordedEdge>>,
}

impl FlightRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start recording the edge behind `slot`, whose negotiated caps are `caps`
    /// and whose producing / consuming elements are named by `label`. Called by
    /// the runner once per edge, after the channels are built and before any
    /// packet flows.
    pub(crate) fn record_edge(&self, label: String, caps: &Caps, slot: &ProbeSlot) {
        let Ok(leading_caps) = encode_packet(&PipelinePacket::CapsChanged(caps.clone())) else {
            crate::g2g_warn!(
                crate::log::Target::category(short_type_name::<Self>()),
                "{label}: caps do not serialize, edge not recorded"
            );
            return;
        };
        let ring = Arc::new(EdgeRing::new(leading_caps));
        slot.install(ring.clone());
        self.edges.lock().push(RecordedEdge { label, ring });
    }

    /// Number of edges being recorded: `0` until a run attaches this recorder,
    /// which is also what a run that was never given it leaves behind.
    pub fn edge_count(&self) -> usize {
        self.edges.lock().len()
    }

    /// Write each recorded edge's retained packets into `dir` as one replayable
    /// recording per edge (`<index>-<producer>-to-<consumer>.g2grec`), leading
    /// with the caps in effect for that edge's oldest retained packet. Returns
    /// the files written, in edge order.
    ///
    /// An edge that carried nothing, or one that carried a device-resident frame
    /// (which cannot be serialized, so cannot be replayed), is skipped with a
    /// warning rather than failing the dump.
    pub fn dump_to_dir(&self, dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>, G2gError> {
        let category = short_type_name::<Self>();
        std::fs::create_dir_all(dir)
            .map_err(|e| crate::log::path_io_err(category, "create", dir, e))?;
        let edges = self.edges.lock();
        let mut written = Vec::new();
        for (index, edge) in edges.iter().enumerate() {
            let (framed, packets) = {
                let state = edge.ring.state.lock();
                if state.unserializable {
                    crate::g2g_warn!(
                        crate::log::Target::category(category),
                        "{}: device memory crossed this edge, not replayable",
                        edge.label
                    );
                    continue;
                }
                if state.entries.is_empty() {
                    continue;
                }
                (state.framed_dump()?, state.entries.len())
            };
            let path = dir.join(format!(
                "{index}-{}.{DUMP_EXTENSION}",
                sanitized(&edge.label)
            ));
            std::fs::write(&path, &framed)
                .map_err(|e| crate::log::path_io_err(category, "write", &path, e))?;
            crate::g2g_info!(
                crate::log::Target::category(category),
                "{}: {packets} packet(s) written to {}",
                edge.label,
                path.display()
            );
            written.push(path);
        }
        Ok(written)
    }
}

/// `label` reduced to characters a file name can hold, so an element named from
/// a launch line cannot steer the dump somewhere else.
fn sanitized(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == NAME_REPLACEMENT {
                c
            } else {
                NAME_REPLACEMENT
            }
        })
        .collect()
}

/// The hop label for an edge from node `src` to node `dst`, using the instance
/// names the runner assigned (a structural node has none, so it is named by
/// index instead).
pub(crate) fn edge_label(names: &[String], src: usize, dst: usize) -> String {
    format!("{}-to-{}", node_label(names, src), node_label(names, dst))
}

fn node_label(names: &[String], node: usize) -> String {
    match names.get(node) {
        Some(name) if !name.is_empty() => name.clone(),
        _ => format!("{UNNAMED_NODE_PREFIX}{node}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::ByteStreamEncoding;
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use crate::wire::decode_packet;

    fn caps(encoding: ByteStreamEncoding) -> Caps {
        Caps::ByteStream { encoding }
    }

    fn frame(sequence: u64, len: usize) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                alloc::vec![sequence as u8; len].into_boxed_slice(),
            )),
            FrameTiming::default(),
            sequence,
        ))
    }

    /// A fresh ring on an edge negotiated as `encoding`.
    fn ring_negotiated_as(encoding: ByteStreamEncoding) -> EdgeRing {
        EdgeRing::new(encode_packet(&PipelinePacket::CapsChanged(caps(encoding))).expect("encodes"))
    }

    /// What a dump of this ring would hold, decoded: the leading caps plus the
    /// retained window.
    fn dumped(ring: &EdgeRing) -> Vec<PipelinePacket> {
        let state = ring.state.lock();
        state
            .records()
            .map(|payload| decode_packet(payload).expect("recorded payloads decode"))
            .collect()
    }

    #[test]
    fn the_packet_bound_keeps_the_most_recent_window() {
        let ring = ring_negotiated_as(ByteStreamEncoding::Ogg);
        let total = FLIGHT_RING_PACKETS as u64 * 2;
        for sequence in 0..total {
            ring.capture(&frame(sequence, 1));
        }
        let records = dumped(&ring);
        assert_eq!(records.len(), FLIGHT_RING_PACKETS + 1, "caps + window");
        let first_kept = match &records[1] {
            PipelinePacket::DataFrame(f) => f.sequence,
            other => panic!("expected a frame, got {other:?}"),
        };
        assert_eq!(first_kept, total - FLIGHT_RING_PACKETS as u64);
    }

    #[test]
    fn the_byte_bound_keeps_at_least_the_newest_packet() {
        let ring = ring_negotiated_as(ByteStreamEncoding::Ogg);
        ring.capture(&frame(0, 8));
        ring.capture(&frame(1, FLIGHT_RING_BYTES + 1));
        let records = dumped(&ring);
        assert_eq!(records.len(), 2, "caps + the oversized newest packet");
        match &records[1] {
            PipelinePacket::DataFrame(f) => assert_eq!(f.sequence, 1),
            other => panic!("expected a frame, got {other:?}"),
        }
    }

    /// A caps change that scrolled out of the window becomes the dump's leading
    /// record: replaying the retained frames under the *negotiated* caps would
    /// type them wrong.
    #[test]
    fn an_evicted_caps_change_becomes_the_leading_record() {
        let ring = ring_negotiated_as(ByteStreamEncoding::Ogg);
        ring.capture(&PipelinePacket::CapsChanged(caps(
            ByteStreamEncoding::Matroska,
        )));
        for sequence in 0..FLIGHT_RING_PACKETS as u64 * 2 {
            ring.capture(&frame(sequence, 1));
        }
        let records = dumped(&ring);
        match &records[0] {
            PipelinePacket::CapsChanged(c) => assert_eq!(
                *c,
                caps(ByteStreamEncoding::Matroska),
                "the change that scrolled out leads the dump"
            ),
            other => panic!("expected leading caps, got {other:?}"),
        }
        assert!(
            !records[1..]
                .iter()
                .any(|p| matches!(p, PipelinePacket::CapsChanged(_))),
            "it is not also inside the window"
        );
    }

    #[test]
    fn a_device_frame_marks_the_edge_unreplayable() {
        let ring = ring_negotiated_as(ByteStreamEncoding::Ogg);
        // SAFETY: fd -1 is never a live DMABUF; `from_raw` only stores it (no
        // I/O) and the Drop `close(-1)` is a harmless no-op. This exercises the
        // capture refusal of a device domain, not real DMABUF handling.
        let dmabuf = unsafe { crate::memory::OwnedDmaBuf::from_raw(-1, 0, 0) };
        ring.capture(&PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::DmaBuf(dmabuf),
            FrameTiming::default(),
            0,
        )));
        assert!(ring.state.lock().unserializable);
        assert!(ring.state.lock().entries.is_empty());
    }

    #[test]
    fn a_name_from_a_launch_line_cannot_steer_the_dump_path() {
        assert_eq!(
            sanitized("../../etc/passwd-to-sink0"),
            ".._.._etc_passwd-to-sink0"
        );
        assert_eq!(edge_label(&[String::from("src0")], 0, 1), "src0-to-node1");
    }
}
