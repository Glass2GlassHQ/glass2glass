//! Fan-out primitives for the dynamic graph layer (DESIGN.md §4.8.4).
//!
//! M9 (1→N slice): a multi-output sink abstraction plus the two routing
//! primitives that cover branch enable/disable and A/B switching:
//!
//! - [`Gate`] — 1→1. Forwards or drops each `DataFrame` by an atomic flag.
//!   It is a plain [`AsyncElement`], so it drops into the existing
//!   `run_source_transform_sink` runner unchanged.
//! - [`Router`] — 1→N. Sends each `DataFrame` to exactly one output port
//!   chosen by an atomic discriminator, and broadcasts `CapsChanged` to
//!   every port. It implements [`MultiOutputElement`], driven by the
//!   `run_source_fanout` runner.
//!
//! Both expose a cloneable control handle ([`GateHandle`], [`RouterHandle`]),
//! mirroring `SwapHandle` (`slot.rs`), so application code or another task
//! flips routing mid-stream without stalling the pipeline.
//!
//! The Merger (fan-in) and `BranchSlot` are a later slice. EOS broadcast on
//! the `Router` is the runner's responsibility, matching the existing
//! "runner forwards Eos" transform contract.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::caps::Caps;
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, ElementBound, OutputSink, PushOutcome,
};
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::runtime::SenderSink;

/// Downstream output addressing one of N ports. The fan-out analog of
/// [`OutputSink`]: `push_to` selects the destination port. Dyn-safe via a
/// boxed future so [`MultiOutputElement`] can take `&mut dyn MultiOutputSink`.
pub trait MultiOutputSink {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>>;

    fn port_count(&self) -> usize;
}

/// [`MultiOutputSink`] backed by one [`SenderSink`] per output link. Built
/// by the fan-out runner from the branch links; `push_to` forwards to the
/// addressed branch.
#[derive(Debug)]
pub struct MultiSenderSink {
    ports: Vec<SenderSink>,
}

impl MultiSenderSink {
    pub fn new(ports: Vec<SenderSink>) -> Self {
        Self { ports }
    }
}

impl MultiOutputSink for MultiSenderSink {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        // Port range is an internal invariant: `Router` clamps its selection
        // and broadcasts only over `0..port_count`, so an out-of-range port
        // is a framework bug, not a runtime error.
        let sink = self.ports.get_mut(port).expect("push_to: port out of range");
        sink.push(packet)
    }

    fn port_count(&self) -> usize {
        self.ports.len()
    }
}

/// Multi-output element trait variant: identical negotiation to
/// [`AsyncElement`], but `process` emits into a [`MultiOutputSink`] rather
/// than a single downstream. [`Router`] is the first implementor; user code
/// can write others (e.g. a content-based demux).
pub trait MultiOutputElement: ElementBound {
    type ProcessFuture<'a>: core::future::Future<Output = Result<(), G2gError>> + 'a
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a>;
}

/// Multi-input element trait variant: an N-input, 1-output element (a
/// muxer). The mirror of [`MultiOutputElement`]. Negotiation is **per
/// input** — each input pad narrows and fixates its own caps — and the
/// element exposes a single merged `output_caps`. The fan-in runner
/// (`run_muxer_sink`) aggregates EOS itself, so `process` is only ever
/// handed `DataFrame`/`CapsChanged`, tagged with the originating `input`.
pub trait MultiInputElement: ElementBound {
    type ProcessFuture<'a>: core::future::Future<Output = Result<(), G2gError>> + 'a
    where
        Self: 'a;

    fn input_count(&self) -> usize;

    /// Phase 1 for one input pad: narrow that input's proposed caps.
    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    /// Phase 2 for one input pad: fixate and configure that input.
    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    /// The merged-output caps, valid once every input has been configured.
    fn output_caps(&self) -> Result<Caps, G2gError>;

    /// Combine one packet from `input` into the merged output.
    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a>;
}

/// 1→1 enable/disable element. Forwards `CapsChanged` unconditionally and
/// `DataFrame` only while open; `Eos` is forwarded by the runner, never by
/// the element (the transform contract). Drops dropped frames silently —
/// observability of gate drops is a tracing concern for a later milestone.
#[derive(Debug)]
pub struct Gate {
    open: Arc<AtomicBool>,
}

impl Gate {
    pub fn new(open: bool) -> Self {
        Self { open: Arc::new(AtomicBool::new(open)) }
    }

    /// A cloneable handle that flips this gate from another task while the
    /// runner drives it.
    pub fn handle(&self) -> GateHandle {
        GateHandle { open: self.open.clone() }
    }
}

/// Detached control handle for a [`Gate`].
#[derive(Debug, Clone)]
pub struct GateHandle {
    open: Arc<AtomicBool>,
}

impl GateHandle {
    pub fn set_open(&self, open: bool) {
        self.open.store(open, Ordering::SeqCst);
    }

    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::SeqCst)
    }
}

impl AsyncElement for Gate {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let open = self.open.load(Ordering::SeqCst);
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if open {
                        out.push(PipelinePacket::DataFrame(f)).await?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                // Runner forwards Eos after process() returns.
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

/// 1→N router. Each `DataFrame` goes to the single port named by an atomic
/// discriminator; `CapsChanged` is broadcast to every port so all branches
/// stay configured. `Eos` is broadcast by the runner.
#[derive(Debug)]
pub struct Router {
    selected: Arc<AtomicUsize>,
    ports: usize,
}

impl Router {
    pub fn new(ports: usize) -> Self {
        assert!(ports > 0, "Router needs at least one output port");
        Self { selected: Arc::new(AtomicUsize::new(0)), ports }
    }

    /// Number of output ports. The fan-out runner allocates one branch link
    /// per port.
    pub fn port_count(&self) -> usize {
        self.ports
    }

    /// A cloneable handle that re-targets this router from another task.
    pub fn handle(&self) -> RouterHandle {
        RouterHandle { selected: self.selected.clone(), ports: self.ports }
    }
}

/// Detached control handle for a [`Router`].
#[derive(Debug, Clone)]
pub struct RouterHandle {
    selected: Arc<AtomicUsize>,
    ports: usize,
}

impl RouterHandle {
    /// Select the output port subsequent `DataFrame`s route to. Panics if
    /// `port >= port_count`.
    pub fn select(&self, port: usize) {
        assert!(port < self.ports, "select: port out of range");
        self.selected.store(port, Ordering::SeqCst);
    }

    pub fn selected(&self) -> usize {
        self.selected.load(Ordering::SeqCst)
    }
}

impl MultiOutputElement for Router {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        // Clamp defensively so a stale handle write can never index past the
        // port list (the runner allocated exactly `ports` branches).
        let selected = self.selected.load(Ordering::SeqCst).min(self.ports - 1);
        let ports = self.ports;
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    out.push_to(selected, PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::CapsChanged(c.clone())).await?;
                    }
                }
                // Runner broadcasts Eos to all ports after process() returns.
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

/// N→1 fan-in selector: the control-driven mirror of [`Router`]. An atomic
/// discriminator names the single active input; the fan-in runner forwards
/// that input's frames and drains/discards the rest. The merged stream ends
/// only once every input has reached EOS (see `run_fanin_sink`). `Merger`
/// holds just the selector; the forwarding lives in the runner.
#[derive(Debug)]
pub struct Merger {
    selected: Arc<AtomicUsize>,
    inputs: usize,
}

impl Merger {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "Merger needs at least one input");
        Self { selected: Arc::new(AtomicUsize::new(0)), inputs }
    }

    /// Number of input ports. The fan-in runner allocates one branch link
    /// per input.
    pub fn input_count(&self) -> usize {
        self.inputs
    }

    /// A cloneable handle that re-selects the active input from another task.
    pub fn handle(&self) -> MergerHandle {
        MergerHandle { selected: self.selected.clone(), inputs: self.inputs }
    }
}

/// Detached control handle for a [`Merger`].
#[derive(Debug, Clone)]
pub struct MergerHandle {
    selected: Arc<AtomicUsize>,
    inputs: usize,
}

impl MergerHandle {
    /// Select which input feeds the merged output. Panics if
    /// `input >= input_count`.
    pub fn select(&self, input: usize) {
        assert!(input < self.inputs, "select: input out of range");
        self.selected.store(input, Ordering::SeqCst);
    }

    pub fn selected(&self) -> usize {
        self.selected.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, VideoFormat};
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use core::future::Future;
    use core::pin::Pin;

    fn caps() -> Caps {
        Caps::Video {
            format: VideoFormat::Rgba8,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn data(seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            caps: caps(),
            timing: FrameTiming { pts_ns: 0, dts_ns: 0, duration_ns: 0, capture_ns: 0 },
            sequence: seq,
        })
    }

    /// Records the kind of every packet pushed, per port, without channels.
    #[derive(Default)]
    struct RecordingMultiSink {
        ports: usize,
        data_seqs: Vec<Vec<u64>>,
        caps_changes: Vec<usize>,
    }

    impl RecordingMultiSink {
        fn new(ports: usize) -> Self {
            Self { ports, data_seqs: alloc::vec![Vec::new(); ports], caps_changes: alloc::vec![0; ports] }
        }
    }

    impl MultiOutputSink for RecordingMultiSink {
        fn push_to<'a>(
            &'a mut self,
            port: usize,
            packet: PipelinePacket,
        ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
            match packet {
                PipelinePacket::DataFrame(f) => self.data_seqs[port].push(f.sequence),
                PipelinePacket::CapsChanged(_) => self.caps_changes[port] += 1,
                PipelinePacket::Eos => {}
            }
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }

        fn port_count(&self) -> usize {
            self.ports
        }
    }

    /// Records every packet a single-output element forwards.
    #[derive(Default)]
    struct RecordingSink {
        data_seqs: Vec<u64>,
        caps_changes: usize,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
            match packet {
                PipelinePacket::DataFrame(f) => self.data_seqs.push(f.sequence),
                PipelinePacket::CapsChanged(_) => self.caps_changes += 1,
                PipelinePacket::Eos => {}
            }
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    /// Single-poll block_on; all futures here resolve immediately.
    fn block_on<F: Future>(mut fut: F) -> F::Output {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        static VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: VT's hooks never dereference the data pointer.
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) };
        let mut cx = Context::from_waker(&waker);
        // SAFETY: `fut` is pinned to the stack for the duration of this call.
        let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("fanout::tests::block_on saw Pending"),
        }
    }

    #[test]
    fn router_sends_each_frame_to_selected_port() {
        let mut router = Router::new(2);
        let handle = router.handle();
        let mut out = RecordingMultiSink::new(2);

        block_on(router.process(data(0), &mut out)).unwrap(); // port 0
        handle.select(1);
        block_on(router.process(data(1), &mut out)).unwrap(); // port 1
        block_on(router.process(data(2), &mut out)).unwrap(); // port 1 (sticky)
        handle.select(0);
        block_on(router.process(data(3), &mut out)).unwrap(); // port 0

        assert_eq!(out.data_seqs[0], alloc::vec![0, 3]);
        assert_eq!(out.data_seqs[1], alloc::vec![1, 2]);
    }

    #[test]
    fn router_broadcasts_caps_changed_to_all_ports() {
        let mut router = Router::new(3);
        let mut out = RecordingMultiSink::new(3);

        block_on(router.process(PipelinePacket::CapsChanged(caps()), &mut out)).unwrap();

        assert_eq!(out.caps_changes, alloc::vec![1, 1, 1]);
    }

    #[test]
    fn gate_open_forwards_data_closed_drops_it() {
        let gate = Gate::new(true);
        let handle = gate.handle();
        let mut gate = gate;
        let mut out = RecordingSink::default();

        block_on(gate.process(data(0), &mut out)).unwrap(); // open -> pass
        handle.set_open(false);
        block_on(gate.process(data(1), &mut out)).unwrap(); // closed -> drop
        handle.set_open(true);
        block_on(gate.process(data(2), &mut out)).unwrap(); // open -> pass

        assert_eq!(out.data_seqs, alloc::vec![0, 2], "frame 1 dropped while closed");
    }

    #[test]
    fn gate_forwards_caps_changed_regardless_of_open_state() {
        let mut gate = Gate::new(false);
        let mut out = RecordingSink::default();

        block_on(gate.process(PipelinePacket::CapsChanged(caps()), &mut out)).unwrap();

        assert_eq!(out.caps_changes, 1, "CapsChanged forwarded even while closed");
    }

    #[test]
    fn merger_handle_selects_active_input() {
        let merger = Merger::new(3);
        let handle = merger.handle();
        assert_eq!(handle.selected(), 0, "defaults to input 0");
        handle.select(2);
        assert_eq!(handle.selected(), 2);
        assert_eq!(merger.input_count(), 3);
    }

    #[test]
    #[should_panic(expected = "input out of range")]
    fn merger_handle_rejects_out_of_range_input() {
        Merger::new(2).handle().select(2);
    }
}
