//! M18 Phase C muxer — per-input re-solve (MX-1) and input-derived output
//! re-emit (MX-2), driven through `run_muxer_sink`.
//!
//! Before Phase C, a per-input mid-stream `CapsChanged` was forwarded
//! straight to the muxer's output (`InterleaveMux::process` pushes every
//! packet), leaking input-side caps as if they were the merged output.
//! MX-1 instead re-solves the changed input against the muxer's per-input
//! constraint and reconfigures that pad; the input `CapsChanged` is
//! consumed, not leaked. MX-2 re-derives the merged output from the
//! freshly reconfigured inputs and eagerly emits one downstream
//! `CapsChanged` only when the output actually changed.
//!
//! Both run entirely inside the single muxer task (which already owns
//! `&mut mux` and serializes all inputs), so they land without the β
//! coordinator restructure, per DESIGN-M16-workaround3-reconfigure.md
//! §10.4.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::mux::InterleaveMux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: seq,
    }
}

/// Source that emits `before` frames under `initial`, then a mid-stream
/// `CapsChanged(switch_to)`, then `after` frames under `switch_to`, EOS.
struct ReconfigSrc {
    initial: Caps,
    switch_to: Caps,
    before: u64,
    after: u64,
    configured: bool,
}

impl SourceLoop for ReconfigSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.initial.clone()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        assert!(self.configured, "runner must configure source before run");
        let switch_to = self.switch_to.clone();
        let before = self.before;
        let after = self.after;
        Box::pin(async move {
            for i in 0..before {
                out.push(PipelinePacket::DataFrame(frame(i))).await?;
            }
            out.push(PipelinePacket::CapsChanged(switch_to.clone())).await?;
            for j in 0..after {
                out.push(PipelinePacket::DataFrame(frame(before + j)))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(before + after)
        })
    }
}

/// Sink recording the data-frame count and every `CapsChanged` it receives,
/// so a test can see exactly what crossed the muxer's output boundary.
#[derive(Default)]
struct RecordingSink {
    frames: u64,
    caps_changes: Vec<Caps>,
    eos: bool,
}

struct ProbeSink {
    log: Arc<Mutex<RecordingSink>>,
}

impl AsyncElement for ProbeSink {
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
        let log = Arc::clone(&self.log);
        Box::pin(async move {
            let mut g = log.lock().unwrap();
            match packet {
                PipelinePacket::DataFrame(_) => g.frames += 1,
                PipelinePacket::CapsChanged(c) => g.caps_changes.push(c),
                PipelinePacket::Eos => g.eos = true,
                PipelinePacket::Flush | PipelinePacket::Segment(_) => {}
            }
            Ok(())
        })
    }
}

/// A muxer whose merged output is *derived* from input 0's configured caps
/// (output = input 0 geometry, forced to NV12). Models a real muxer whose
/// output format tracks an input, so MX-2 has an observable output change.
struct DerivedMux {
    inputs: usize,
    configured: Vec<Option<Caps>>,
}

impl DerivedMux {
    fn new(inputs: usize) -> Self {
        Self { inputs, configured: vec![None; inputs] }
    }

    fn derived_output(&self) -> Caps {
        match self.configured.first().and_then(|c| c.as_ref()).and_then(|c| c.dims()) {
            Some((width, height, framerate)) => Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            },
            None => nv12(2, 2),
        }
    }
}

impl MultiInputElement for DerivedMux {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.derived_output())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.derived_output())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            out.push(packet).await?;
            Ok(())
        })
    }
}

/// MX-1: a per-input mid-stream `CapsChanged` re-solves and reconfigures
/// that input pad, and is consumed (not leaked downstream). With a static
/// output muxer (`InterleaveMux`) the sink sees no `CapsChanged` at all.
#[tokio::test]
async fn mx1_per_input_capschanged_reconfigures_pad_and_is_not_leaked() {
    let log = Arc::new(Mutex::new(RecordingSink::default()));
    let mut src = ReconfigSrc {
        initial: rgba(640, 480),
        switch_to: rgba(1920, 1080),
        before: 2,
        after: 2,
        configured: false,
    };
    let mut mux = InterleaveMux::new(1, rgba(640, 480));
    let mut snk = ProbeSink { log: Arc::clone(&log) };
    let clock = ZeroClock;

    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
        run_muxer_sink(sources, &mut mux, &mut snk, &clock, 4)
            .await
            .expect("muxer pipeline completes");
    }

    // MX-1 reconfigured input 0 to the new caps.
    assert_eq!(
        mux.input_caps(0),
        Some(&rgba(1920, 1080)),
        "input 0 pad must re-solve to the new caps"
    );

    let g = log.lock().unwrap();
    assert!(g.eos, "EOS reaches the sink");
    assert_eq!(g.frames, 4, "all frames forwarded");
    // Static output: the input CapsChanged was consumed by MX-1, and MX-2
    // saw no output change, so nothing leaked downstream.
    assert!(
        g.caps_changes.is_empty(),
        "input-side CapsChanged must NOT leak to the muxer output (got {:?})",
        g.caps_changes
    );
}

/// MX-2: when the per-input change shifts the derived merged output, the
/// muxer eagerly emits exactly one downstream `CapsChanged` carrying the
/// new output caps.
#[tokio::test]
async fn mx2_input_derived_output_change_emits_one_downstream_capschanged() {
    let log = Arc::new(Mutex::new(RecordingSink::default()));
    let mut src = ReconfigSrc {
        initial: rgba(640, 480),
        switch_to: rgba(1920, 1080),
        before: 2,
        after: 2,
        configured: false,
    };
    let mut mux = DerivedMux::new(1);
    let mut snk = ProbeSink { log: Arc::clone(&log) };
    let clock = ZeroClock;

    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
        run_muxer_sink(sources, &mut mux, &mut snk, &clock, 4)
            .await
            .expect("muxer pipeline completes");
    }

    let g = log.lock().unwrap();
    assert!(g.eos, "EOS reaches the sink");
    assert_eq!(g.frames, 4, "all frames forwarded");
    // The output derives from input 0 geometry: 640x480 -> 1920x1080 NV12.
    // MX-2 emits exactly one downstream CapsChanged with the new output.
    assert_eq!(
        g.caps_changes,
        vec![nv12(1920, 1080)],
        "MX-2 must emit one downstream CapsChanged with the re-derived output"
    );
}
