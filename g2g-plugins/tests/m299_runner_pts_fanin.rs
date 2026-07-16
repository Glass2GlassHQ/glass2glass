//! M299 - runner-level PTS-ordered fan-in. A muxer that does NO internal
//! buffering opts into `MultiInputElement::input_pts_ordered`, and the runner's
//! `muxer_arm_pts` merges its inputs by `DataFrame` PTS: it releases the
//! globally-earliest frame only once every still-open input has one queued, so a
//! non-aggregating element still sees its inputs in timestamp order. This is the
//! capability a multi-camera grid / PTS-synchronized compositor needs without
//! hand-rolling an `InputAggregator` (contrast `m204`, where the *element* does
//! the merge). The output PTS sequence being sorted here proves the *runner*
//! ordered it, since the muxer just forwards whatever it is handed.

use std::pin::Pin;

use core::future::Future;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits frames at the given PTS values (ns), then EOS, yielding between each so
/// the two sources actually interleave their delivery to the runner.
struct PtsSrc {
    pts: Vec<u64>,
    configured: bool,
}

impl SourceLoop for PtsSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let pts = self.pts.clone();
        let configured = self.configured;
        Box::pin(async move {
            assert!(configured, "runner configures the source before run");
            for p in &pts {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![0u8; 4].into_boxed_slice(),
                    )),
                    timing: FrameTiming { pts_ns: *p, ..FrameTiming::default() },
                    sequence: *p,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pts.len() as u64)
        })
    }
}

/// A muxer with NO internal buffering: it forwards each `DataFrame` straight to
/// its output in the exact order `process` is called, and opts into runner-level
/// PTS ordering. Any sorting in the output therefore comes from the runner.
struct PassThroughMux {
    inputs: usize,
}

impl MultiInputElement for PassThroughMux {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn input_pts_ordered(&self) -> bool {
        true
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _input: usize, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(caps())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // Forward data verbatim; never forward Eos (the runner owns the merged
            // one). The runner consumes input CapsChanged itself, so it is not seen.
            if let PipelinePacket::DataFrame(_) = packet {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

/// Records the PTS of each frame in arrival (output) order.
#[derive(Default)]
struct OrderSink {
    pts: Vec<u64>,
}

impl AsyncElement for OrderSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(f) = packet {
            self.pts.push(f.timing.pts_ns);
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn runner_orders_a_non_aggregating_muxers_inputs_by_pts() {
    // A at 0,20,40; B at 10,30,50. The muxer does no buffering, so a sorted
    // output can only be the runner's PTS merge.
    let mut a = PtsSrc { pts: vec![0, 20, 40], configured: false };
    let mut b = PtsSrc { pts: vec![10, 30, 50], configured: false };
    let mut mux = PassThroughMux { inputs: 2 };
    let mut sink = OrderSink::default();

    let stats = {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink(sources, &mut mux, &mut sink, &ZeroClock, 4)
            .await
            .expect("muxer pipeline completes")
    };

    assert_eq!(stats.frames_consumed, 6, "all frames reached the sink");
    assert_eq!(
        sink.pts,
        vec![0, 10, 20, 30, 40, 50],
        "the runner merged the inputs in global PTS order"
    );
}

#[tokio::test]
async fn runner_flushes_pts_fanin_when_one_input_ends_early() {
    // A ends after one frame (5); B continues (10,20,30). Once A ends and drains,
    // B's later frames must still flow (A drops out of the merge round).
    let mut a = PtsSrc { pts: vec![5], configured: false };
    let mut b = PtsSrc { pts: vec![10, 20, 30], configured: false };
    let mut mux = PassThroughMux { inputs: 2 };
    let mut sink = OrderSink::default();

    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink(sources, &mut mux, &mut sink, &ZeroClock, 4)
            .await
            .expect("muxer pipeline completes");
    }

    assert_eq!(sink.pts, vec![5, 10, 20, 30], "earliest first, then B drains after A ends");
}
