//! M348: merged downstream output for dynamic fan-in. `run_aggregator_dynamic`
//! (M320) drives a *terminal* aggregator (merged output discarded);
//! `run_muxer_sink_dynamic` extends it to the `run_muxer_sink` shape, a trailing
//! sink fed the muxer's merged output, with the output caps coupled to the sink
//! as inputs attach at runtime. Pure-fake elements (no hardware).

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::runtime::{
    run_muxer_sink_dynamic, run_muxer_sink_dynamic_observed, DynSourceLoop, NodeRole, Observer,
    SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MultiInputElement,
    OutputSink, PipelinePacket, Rate, RawVideoFormat,
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

/// Interleave muxer: every input frame is forwarded to the merged output (so the
/// trailing sink actually receives data), and the merged output caps are fixed.
struct PassthroughMux {
    inputs: usize,
}

impl MultiInputElement for PassthroughMux {
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
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                // The runner owns the merged EOS; the muxer must not forward it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Sink counting merged frames; records the caps it was configured with so the
/// output-caps coupling is observable (it must be configured before any frame).
struct RecordingSink {
    frames: Arc<Mutex<u64>>,
    configured_with: Arc<Mutex<Option<Caps>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        *self.configured_with.lock().unwrap() = Some(c.clone());
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                assert!(
                    self.configured_with.lock().unwrap().is_some(),
                    "sink must be configured (output CapsChanged) before any merged frame"
                );
                *self.frames.lock().unwrap() += 1;
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn dynamic_inputs_feed_a_trailing_sink_with_coupled_output_caps() {
    let frames = Arc::new(Mutex::new(0u64));
    let configured = Arc::new(Mutex::new(None));
    let mut mux = PassthroughMux { inputs: 3 };
    let mut sink = RecordingSink {
        frames: Arc::clone(&frames),
        configured_with: Arc::clone(&configured),
    };

    let (handle, run) = run_muxer_sink_dynamic(&mut mux, &mut sink, 4);
    handle
        .add_input(Box::new(CountedSource { n: 5 }) as Box<dyn DynSourceLoop>)
        .expect("add input 0");
    handle
        .add_input(Box::new(CountedSource { n: 3 }) as Box<dyn DynSourceLoop>)
        .expect("add input 1");
    drop(handle);

    let stats = run.await.expect("dynamic muxer->sink run");

    assert_eq!(
        *frames.lock().unwrap(),
        8,
        "the sink received every merged frame (5 + 3)"
    );
    assert_eq!(
        stats.frames_consumed, 8,
        "frames_consumed is the sink's merged count"
    );
    assert_eq!(
        stats.frames_emitted, 8,
        "both runtime inputs' frames summed"
    );
    assert_eq!(
        *configured.lock().unwrap(),
        Some(caps()),
        "the muxer's merged output caps were coupled to the sink"
    );
}

/// M869: the observed variant. The muxer and its trailing sink are registered
/// with probes up front, each runtime input appends its node and link, and the
/// merged `muxer -> sink` edge joins the topology once its caps firm up.
#[tokio::test]
async fn observed_dynamic_muxer_sink_reports_every_stage() {
    const N: u64 = 200;
    let frames = Arc::new(Mutex::new(0u64));
    let configured = Arc::new(Mutex::new(None));
    let mut mux = PassthroughMux { inputs: 3 };
    let mut sink = RecordingSink {
        frames: Arc::clone(&frames),
        configured_with: Arc::clone(&configured),
    };

    let obs = Observer::new();
    let (handle, run) = run_muxer_sink_dynamic_observed(&mut mux, &mut sink, 2, &obs);
    handle
        .add_input(Box::new(CountedSource { n: N }) as Box<dyn DynSourceLoop>)
        .expect("add input 0");

    let late = {
        let obs = obs.clone();
        async move {
            let mut before = 0usize;
            for _ in 0..1_000_000 {
                before = obs.snapshot().nodes.len();
                if before == 3 {
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
    let stats = stats.expect("observed dynamic muxer->sink run");
    assert_eq!(
        nodes_before, 3,
        "muxer + sink + first input were visible mid-run"
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
            ("PassthroughMux0", NodeRole::Muxer),
            ("RecordingSink0", NodeRole::Sink),
            ("CountedSource0", NodeRole::Source),
            ("CountedSource1", NodeRole::Source),
        ],
    );
    // Both inputs feed the muxer, and the merged link reaches the sink. Order is
    // registration order (an input on attach, the merged link once its caps
    // solve), so compare the set.
    let mut pairs = snap
        .edges
        .iter()
        .map(|e| (e.from, e.to))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(0, 1), (2, 0), (3, 0)]);
    assert!(
        snap.edges.iter().all(|e| e.caps.is_some()),
        "every edge carries its caps, the merged one included"
    );
    let merged = snap
        .edges
        .iter()
        .find(|e| (e.from, e.to) == (0, 1))
        .expect("the merged muxer -> sink link is registered");
    assert!(
        merged.counts.packets > 0,
        "the merged link counted traffic: {:?}",
        merged.counts
    );

    // Both stages with a `process()` report measured rows.
    assert_eq!(
        stats
            .per_element
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>(),
        vec!["PassthroughMux0", "RecordingSink0"],
    );
    assert_eq!(
        stats.per_element[0].proc.count,
        2 * N,
        "the muxer timed every input frame"
    );
    assert_eq!(
        stats.per_element[1].proc.count,
        2 * N,
        "the sink timed every merged frame"
    );
    assert_eq!(*frames.lock().unwrap(), 2 * N);
}
