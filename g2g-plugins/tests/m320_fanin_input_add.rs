//! M320: fan-in request pads. An input is attached to a *running* aggregator via
//! `DynamicFaninHandle::add_input` (the runtime equivalent of GStreamer's
//! aggregator/muxer request **sink** pads), the dual of the M310/M319 dynamic
//! fan-out. The aggregator declares a fixed pad capacity; each attached source
//! self-fixates, its pad is configured on attach, and its frames are tagged with
//! the pad index. The run ends once the handle is dropped and every attached
//! input has reached EOS.

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;

use g2g_core::runtime::{
    run_aggregator_dynamic, run_aggregator_dynamic_observed, DynSourceLoop, NodeRole, Observer,
    SourceLoop,
};
use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MultiInputElement, OutputSink,
    PipelinePacket, Rate, RawVideoFormat,
};

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// Source pushing `n` frames then EOS.
struct CountedSource {
    n: u64,
}

impl SourceLoop for CountedSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.n {
                let frame = g2g_core::frame::Frame::new(
                    g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                        std::vec![0u8; 4].into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming {
                        pts_ns: seq,
                        ..Default::default()
                    },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// Terminal aggregator: records frames + EOS per pad. `inputs` is the pad
/// capacity (the dynamic handle hands out pads `0..inputs`).
struct RecordingAggregator {
    inputs: usize,
    frames: std::vec::Vec<u64>,
    eos: std::vec::Vec<u64>,
}

impl RecordingAggregator {
    fn new(inputs: usize) -> Self {
        Self {
            inputs,
            frames: std::vec![0; inputs],
            eos: std::vec![0; inputs],
        }
    }
}

impl MultiInputElement for RecordingAggregator {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(_) => self.frames[input] += 1,
                PipelinePacket::Eos => self.eos[input] += 1,
                _ => {}
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn runtime_inputs_attach_to_distinct_pads_and_end_on_all_eos() {
    // Pad capacity 3; two inputs added at runtime take pads 0 and 1, the third
    // stays a dark pad (no source ever attaches).
    let mut agg = RecordingAggregator::new(3);
    let (handle, run) = run_aggregator_dynamic(&mut agg, 4);

    // Request two sink pads before the run is driven: they queue on the control
    // channel and are folded in on the first aggregator poll.
    handle
        .add_input(Box::new(CountedSource { n: 5 }) as Box<dyn DynSourceLoop>)
        .expect("add input 0");
    handle
        .add_input(Box::new(CountedSource { n: 3 }) as Box<dyn DynSourceLoop>)
        .expect("add input 1");
    // Dropping the handle stops accepting new inputs; the two queued still attach,
    // and the run ends once both have sent EOS.
    drop(handle);

    let stats = run.await.expect("dynamic fan-in run");

    // Each input's frames landed on its own pad, in add order; pad 2 stayed dark.
    assert_eq!(
        agg.frames,
        std::vec![5, 3, 0],
        "per-pad frame routing for runtime inputs"
    );
    assert_eq!(
        agg.eos,
        std::vec![1, 1, 0],
        "per-input EOS delivered to its pad"
    );
    assert_eq!(
        stats.frames_consumed, 8,
        "aggregator consumed the union of inputs"
    );
    assert_eq!(
        stats.frames_emitted, 8,
        "both runtime inputs' frames summed"
    );
}

/// M869: the dual of the observed dynamic fan-out. The aggregator is registered
/// with its measured-latency probe up front; each runtime input appends its own
/// node and per-input link, so an input attached mid-run is visible in a snapshot
/// with its traffic counters, and the run reports the aggregator's row.
#[tokio::test]
async fn observed_dynamic_fanin_reports_late_input_telemetry() {
    const N: u64 = 200;
    let mut agg = RecordingAggregator::new(3);
    let obs = Observer::new();
    // Depth 2 so the sources back-pressure and the run spans many polls.
    let (handle, run) = run_aggregator_dynamic_observed(&mut agg, 2, &obs);
    handle
        .add_input(Box::new(CountedSource { n: N }) as Box<dyn DynSourceLoop>)
        .expect("add input 0");

    // Attach the second input once the first is in the topology, i.e. after the
    // aggregator has started draining.
    let late = {
        let obs = obs.clone();
        async move {
            let mut before = 0usize;
            for _ in 0..1_000_000 {
                before = obs.snapshot().nodes.len();
                if before == 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            let added = handle
                .add_input(Box::new(CountedSource { n: N }) as Box<dyn DynSourceLoop>)
                .is_ok();
            drop(handle);
            (before, added)
        }
    };

    let (stats, (nodes_before, added)) = tokio::join!(run, late);
    let stats = stats.expect("observed dynamic fan-in run");
    assert_eq!(
        nodes_before, 2,
        "aggregator + first input were visible mid-run"
    );
    assert!(
        added,
        "the second input attached while the run was in flight"
    );

    let snap = obs.snapshot();
    assert_eq!(
        snap.nodes
            .iter()
            .map(|n| (n.name.as_str(), n.role))
            .collect::<Vec<_>>(),
        vec![
            ("RecordingAggregator0", NodeRole::Muxer),
            ("CountedSource0", NodeRole::Source),
            ("CountedSource1", NodeRole::Source),
        ],
        "the late input grew the node list, named like a built-in one"
    );
    // Each input's link points at the aggregator and carries caps + counts.
    assert_eq!(
        snap.edges
            .iter()
            .map(|e| (e.from, e.to))
            .collect::<Vec<_>>(),
        vec![(1, 0), (2, 0)],
    );
    assert!(
        snap.edges.iter().all(|e| e.caps.is_some()),
        "both input edges carry their fixated caps"
    );
    assert!(
        snap.edges[1].counts.packets > 0,
        "the late input's link counted traffic: {:?}",
        snap.edges[1].counts
    );

    // The aggregator reports a measured row, no longer an empty `per_element`.
    assert_eq!(stats.per_element.len(), 1);
    let row = &stats.per_element[0];
    assert_eq!(row.name, "RecordingAggregator0");
    assert_eq!(
        row.proc.count,
        2 * N,
        "every aggregated frame was timed at the aggregator"
    );
    assert!(
        row.fill_max_pct > 0,
        "the aggregator sampled its input fill, max={}",
        row.fill_max_pct
    );
    assert_eq!(agg.frames, std::vec![N, N, 0]);
}

#[tokio::test]
async fn add_input_past_pad_capacity_is_rejected() {
    // Capacity 1: the first input attaches, the second has no free pad.
    let mut agg = RecordingAggregator::new(1);
    let (handle, run) = run_aggregator_dynamic(&mut agg, 4);

    handle
        .add_input(Box::new(CountedSource { n: 4 }) as Box<dyn DynSourceLoop>)
        .expect("first input fits the single pad");
    let rejected = handle.add_input(Box::new(CountedSource { n: 9 }) as Box<dyn DynSourceLoop>);
    assert!(
        rejected.is_err(),
        "no free pad: the second add must be rejected"
    );
    drop(handle);

    let stats = run
        .await
        .expect("run completes with the one attached input");
    assert_eq!(
        agg.frames,
        std::vec![4],
        "only the accepted input's frames were aggregated"
    );
    assert_eq!(stats.frames_consumed, 4);
}
