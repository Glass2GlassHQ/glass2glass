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

use g2g_core::runtime::{run_aggregator_dynamic, DynSourceLoop, SourceLoop};
use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MultiInputElement, OutputSink,
    PipelinePacket, RawVideoFormat, Rate,
};

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
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
                    g2g_core::FrameTiming { pts_ns: seq, ..Default::default() },
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
        Self { inputs, frames: std::vec![0; inputs], eos: std::vec![0; inputs] }
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
    assert_eq!(agg.frames, std::vec![5, 3, 0], "per-pad frame routing for runtime inputs");
    assert_eq!(agg.eos, std::vec![1, 1, 0], "per-input EOS delivered to its pad");
    assert_eq!(stats.frames_consumed, 8, "aggregator consumed the union of inputs");
    assert_eq!(stats.frames_emitted, 8, "both runtime inputs' frames summed");
}

#[tokio::test]
async fn add_input_past_pad_capacity_is_rejected() {
    // Capacity 1: the first input attaches, the second has no free pad.
    let mut agg = RecordingAggregator::new(1);
    let (handle, run) = run_aggregator_dynamic(&mut agg, 4);

    handle
        .add_input(Box::new(CountedSource { n: 4 }) as Box<dyn DynSourceLoop>)
        .expect("first input fits the single pad");
    let rejected =
        handle.add_input(Box::new(CountedSource { n: 9 }) as Box<dyn DynSourceLoop>);
    assert!(rejected.is_err(), "no free pad: the second add must be rejected");
    drop(handle);

    let stats = run.await.expect("run completes with the one attached input");
    assert_eq!(agg.frames, std::vec![4], "only the accepted input's frames were aggregated");
    assert_eq!(stats.frames_consumed, 4);
}
