use core::convert::Infallible;
use core::task::Poll;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use spin::Mutex;

use crate::element::BoxFuture;
use crate::error::G2gError;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::log::{Target, RUNTIME_CATEGORY};
use crate::runtime::seek::seek_target;
use crate::runtime::{GraphNodeRef, SeekController};

#[derive(Debug, Default, Clone, Copy)]
struct SinkStreamEnd {
    flushes: u64,
    at_eos: bool,
}

#[derive(Debug)]
struct SeekableRun {
    controller: SeekController,
    sinks: Vec<Mutex<SinkStreamEnd>>,
}

impl SeekableRun {
    // a sink's Eos belongs to the latest seek once it has seen that seek's Flush
    fn every_sink_ended(&self, flushing_seeks_taken: u64) -> bool {
        self.sinks.iter().all(|sink| {
            let sink = *sink.lock();
            sink.at_eos && sink.flushes >= flushing_seeks_taken
        })
    }
}

// what a sink downstream of the seekable node reports, so the run can tell when nothing more is sought
#[derive(Debug, Clone)]
pub(crate) struct SinkEndReport {
    run: Arc<SeekableRun>,
    slot: usize,
}

impl SinkEndReport {
    pub(crate) fn flushed(&self) {
        let mut sink = self.run.sinks[self.slot].lock();
        sink.flushes += 1;
        sink.at_eos = false;
    }

    pub(crate) fn eos_is_stale(&self) -> bool {
        self.run.sinks[self.slot].lock().flushes < self.run.controller.flushing_seeks_taken()
    }

    pub(crate) fn ended(&self) {
        self.run.sinks[self.slot].lock().at_eos = true;
        self.run.controller.wake_run_end();
    }
}

// the Flushes an arm downstream of the seekable node has seen
#[derive(Debug)]
pub(crate) struct ArmFlushCount {
    controller: SeekController,
    flushes: u64,
}

impl ArmFlushCount {
    pub(crate) fn flushed(&mut self) {
        self.flushes += 1;
    }

    // an Eos sent before the Flush of a seek already taken, the stream continues after that Flush
    pub(crate) fn eos_is_stale(&self) -> bool {
        self.flushes < self.controller.flushing_seeks_taken()
    }
}

// a run whose seekable source parks at the end of the stream and takes seeks there
#[derive(Debug)]
pub(crate) struct SeekAfterEnd {
    run: Arc<SeekableRun>,
    // by node index: the seekable node, its byte source, and everything downstream
    participants: Vec<bool>,
    sink_slots: Vec<Option<usize>>,
}

impl SeekAfterEnd {
    // None when the pipeline holds a muxer, a fan-in or a sink that finalizes its output at Eos
    pub(crate) fn plan(
        graph: &Graph<GraphNodeRef<'_>>,
        controller: &SeekController,
    ) -> Result<Option<Self>, G2gError> {
        let target = seek_target(graph).map_err(|error| {
            crate::g2g_error!(
                Target::category(RUNTIME_CATEGORY),
                "the run was handed a seek controller, but {error}"
            );
            G2gError::SeekRefused
        })?;
        let node_count = graph.node_count();
        let nodes = || (0..node_count).map(|index| NodeId(index as u32));
        let has_fan_in = nodes().any(|node| {
            matches!(
                graph.node_kind(node),
                Some(NodeKind::Muxer(_) | NodeKind::FaninSink(_))
            )
        });
        let has_finalizing_sink = nodes().any(|node| match graph.element(node) {
            Some(GraphNodeRef::Element(element)) => element.finalizes_on_eos(),
            _ => false,
        });
        if has_fan_in || has_finalizing_sink {
            crate::g2g_info!(
                Target::category(RUNTIME_CATEGORY),
                "the pipeline has a muxer, a fan-in or a sink that finalizes at Eos, so the run ends at the end of the stream and takes no seek after it"
            );
            return Ok(None);
        }
        let mut participants = alloc::vec![false; node_count];
        let mut reached = alloc::vec![target.time_node()];
        while let Some(node) = reached.pop() {
            if core::mem::replace(&mut participants[node.0 as usize], true) {
                continue;
            }
            reached.extend(
                graph
                    .edges()
                    .iter()
                    .filter(|edge| edge.src.node == node)
                    .map(|edge| edge.dst.node),
            );
        }
        if let Some(byte_source) = target.byte_source() {
            participants[byte_source.0 as usize] = true;
        }
        let mut sink_slots = alloc::vec![None; node_count];
        let mut sink_count = 0;
        for node in nodes() {
            if participants[node.0 as usize] && graph.node_kind(node) == Some(NodeKind::Sink) {
                sink_slots[node.0 as usize] = Some(sink_count);
                sink_count += 1;
            }
        }
        controller.enable_seeks_after_end();
        Ok(Some(Self {
            run: Arc::new(SeekableRun {
                controller: controller.clone(),
                sinks: (0..sink_count).map(|_| Mutex::default()).collect(),
            }),
            participants,
            sink_slots,
        }))
    }

    fn participates(&self, node: NodeId) -> bool {
        self.participants[node.0 as usize]
    }

    pub(crate) fn arm_flush_count(&self, node: NodeId) -> Option<ArmFlushCount> {
        self.participates(node).then(|| ArmFlushCount {
            controller: self.run.controller.clone(),
            flushes: 0,
        })
    }

    pub(crate) fn sink_report(&self, node: NodeId) -> Option<SinkEndReport> {
        self.sink_slots[node.0 as usize].map(|slot| SinkEndReport {
            run: self.run.clone(),
            slot,
        })
    }

    // an arm that ends before the source returns cannot carry a later seek, so its end stops the parked source
    pub(crate) fn controller_ended_with(&self, node: NodeId) -> Option<SeekController> {
        self.participates(node).then(|| self.run.controller.clone())
    }

    pub(crate) fn end_when_nothing_more_is_sought(
        &self,
    ) -> impl core::future::Future<Output = Infallible> {
        let run = self.run.clone();
        core::future::poll_fn(move |cx| {
            let controller = &run.controller;
            controller.register_run_end_waker(cx.waker());
            let flushing_seeks_taken = controller.flushing_seeks_taken();
            if run.every_sink_ended(flushing_seeks_taken) && !controller.has_pending() {
                controller.mark_stream_ended(flushing_seeks_taken);
                if !controller.run_held_open() {
                    controller.shutdown();
                }
            }
            Poll::Pending
        })
    }
}

pub(crate) fn shut_down_when_done<'f>(
    arm: BoxFuture<'f, Result<u64, G2gError>>,
    controller: Option<SeekController>,
) -> BoxFuture<'f, Result<u64, G2gError>> {
    match controller {
        Some(controller) => Box::pin(async move {
            let result = arm.await;
            controller.shutdown();
            result
        }),
        None => arm,
    }
}
