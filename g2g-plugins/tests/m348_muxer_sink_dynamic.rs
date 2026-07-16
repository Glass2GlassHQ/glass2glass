//! M348: merged downstream output for dynamic fan-in. `run_aggregator_dynamic`
//! (M320) drives a *terminal* aggregator (merged output discarded);
//! `run_muxer_sink_dynamic` extends it to the `run_muxer_sink` shape, a trailing
//! sink fed the muxer's merged output, with the output caps coupled to the sink
//! as inputs attach at runtime. Pure-fake elements (no hardware).

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::runtime::{run_muxer_sink_dynamic, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MultiInputElement,
    OutputSink, PipelinePacket, RawVideoFormat, Rate,
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
    let mut sink =
        RecordingSink { frames: Arc::clone(&frames), configured_with: Arc::clone(&configured) };

    let (handle, run) = run_muxer_sink_dynamic(&mut mux, &mut sink, 4);
    handle
        .add_input(Box::new(CountedSource { n: 5 }) as Box<dyn DynSourceLoop>)
        .expect("add input 0");
    handle
        .add_input(Box::new(CountedSource { n: 3 }) as Box<dyn DynSourceLoop>)
        .expect("add input 1");
    drop(handle);

    let stats = run.await.expect("dynamic muxer->sink run");

    assert_eq!(*frames.lock().unwrap(), 8, "the sink received every merged frame (5 + 3)");
    assert_eq!(stats.frames_consumed, 8, "frames_consumed is the sink's merged count");
    assert_eq!(stats.frames_emitted, 8, "both runtime inputs' frames summed");
    assert_eq!(
        *configured.lock().unwrap(),
        Some(caps()),
        "the muxer's merged output caps were coupled to the sink"
    );
}
