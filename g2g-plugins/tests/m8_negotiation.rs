//! M8: tests for the renegotiation runner primitives.
//!
//! Piece 1 — runner-driven `CapsChanged` cascade: when a source pushes a
//! `CapsChanged` packet mid-stream, the runner must call
//! `configure_pipeline()` on every downstream element before any
//! subsequent `DataFrame` reaches it.

use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, PushOutcome, Rate, Reconfigure, VideoCodec, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps_at(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source that emits a deterministic packet pattern:
///   DataFrame(0, caps=initial), CapsChanged(refined), DataFrame(1, caps=refined), Eos
struct CapsChangingTestSrc {
    initial: Caps,
    refined: Caps,
    configured: bool,
}

impl SourceLoop for CapsChangingTestSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.initial.clone()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            assert!(self.configured, "runner must configure source before run");

            let refined = self.refined.clone();

            out.push(PipelinePacket::DataFrame(make_frame(0)))
                .await?;
            out.push(PipelinePacket::CapsChanged(refined.clone()))
                .await?;
            out.push(PipelinePacket::DataFrame(make_frame(1)))
                .await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(2)
        })
    }
}

fn make_frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: seq,
    }
}

/// Sink that records, in order of arrival, every `configure_pipeline`
/// invocation and every `process()` invocation with its packet kind.
/// Lets tests assert exact ordering between caps reconfiguration and
/// data delivery.
#[derive(Debug, Default)]
struct OrderedRecordingSink {
    log: Mutex<Vec<Event>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    Configure { width: u32 },
    Data { seq: u64 },
    CapsChanged { width: u32 },
    Flush,
    Eos,
}

impl OrderedRecordingSink {
    fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }

    fn push(&self, e: Event) {
        self.log.lock().unwrap().push(e);
    }
}

impl AsyncElement for OrderedRecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let width = match absolute_caps {
            Caps::RawVideo { width: Dim::Fixed(w), .. } => *w,
            _ => 0,
        };
        self.push(Event::Configure { width });
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let event = match &packet {
            PipelinePacket::DataFrame(f) => Event::Data { seq: f.sequence },
            PipelinePacket::CapsChanged(Caps::RawVideo { width: Dim::Fixed(w), .. }) => {
                Event::CapsChanged { width: *w }
            }
            PipelinePacket::CapsChanged(_) => Event::CapsChanged { width: 0 },
            PipelinePacket::Flush => Event::Flush,
            PipelinePacket::Eos => Event::Eos,
        };
        self.push(event);
        Box::pin(async { Ok(()) })
    }
}

/// Sink that returns `ReFixate(counter)` from the first
/// `configure_pipeline()` call and `Accepted` thereafter. Used to
/// exercise the Phase 3 bounded-retry path.
struct RefixateOnceSink {
    counter: Caps,
    configures: usize,
    last_accepted_caps: Option<Caps>,
}

impl RefixateOnceSink {
    fn new(counter: Caps) -> Self {
        Self { counter, configures: 0, last_accepted_caps: None }
    }
}

impl AsyncElement for RefixateOnceSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configures += 1;
        if self.configures == 1 {
            Ok(ConfigureOutcome::ReFixate(self.counter.clone()))
        } else {
            self.last_accepted_caps = Some(absolute_caps.clone());
            Ok(ConfigureOutcome::Accepted)
        }
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Minimal source for Piece 5 testing: advertises one caps, accepts whatever
/// fixated caps come back, then emits a single Eos. The point is to drive
/// the runner's negotiation phase without contributing data complexity.
struct StaticCapsSrc {
    proposal: Caps,
}

impl SourceLoop for StaticCapsSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.proposal.clone()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

#[tokio::test]
async fn phase3_refixate_restarts_negotiation_with_counter() {
    let mut src = StaticCapsSrc { proposal: caps_at(640, 480) };
    let counter = caps_at(320, 240);
    let mut snk = RefixateOnceSink::new(counter.clone());
    let clock = ZeroClock;

    run_simple_pipeline(&mut src, &mut snk, &clock, 4)
        .await
        .expect("bounded retry must converge");

    assert_eq!(snk.configures, 2, "sink must see two configure attempts");
    assert_eq!(
        snk.last_accepted_caps.as_ref(),
        Some(&counter),
        "second attempt must use the sink's counter-proposal"
    );
}

/// Sink that always returns `ReFixate(same_counter)`. Exists to verify
/// the runner gives up after `MAX_FIXATION_ATTEMPTS`.
struct AlwaysRefixateSink {
    counter: Caps,
}

impl AsyncElement for AlwaysRefixateSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::ReFixate(self.counter.clone()))
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn phase3_refixate_gives_up_after_bounded_attempts() {
    let mut src = StaticCapsSrc { proposal: caps_at(640, 480) };
    let mut snk = AlwaysRefixateSink { counter: caps_at(320, 240) };
    let clock = ZeroClock;

    let err = run_simple_pipeline(&mut src, &mut snk, &clock, 4)
        .await
        .expect_err("infinite refixate must fail");
    assert_eq!(err, G2gError::FixationFailed);
}

/// Sink that accepts caps where width == `accept_width`; for any other
/// caps it returns a single `ReFixate(counter)` per absolute-caps value,
/// then accepts. Combined with [`ReconfigurableTestSrc`] this exercises
/// the full Reconfigure round-trip: source emits CapsChanged → sink
/// rejects → runner fires Reconfigure upstream → source reacts via
/// `reconfigure()` → source emits new CapsChanged → sink accepts.
struct PickyByWidthSink {
    accept_width: u32,
    counter: Caps,
    configures: Mutex<Vec<u32>>, // widths seen, in order
    current_width: Mutex<u32>,
    received_data_widths: Mutex<Vec<u32>>,
}

impl PickyByWidthSink {
    fn new(accept_width: u32, counter: Caps) -> Self {
        Self {
            accept_width,
            counter,
            configures: Mutex::new(Vec::new()),
            current_width: Mutex::new(0),
            received_data_widths: Mutex::new(Vec::new()),
        }
    }

    fn configure_widths(&self) -> Vec<u32> {
        self.configures.lock().unwrap().clone()
    }

    fn data_widths(&self) -> Vec<u32> {
        self.received_data_widths.lock().unwrap().clone()
    }
}

impl AsyncElement for PickyByWidthSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let w = match absolute_caps {
            Caps::RawVideo { width: Dim::Fixed(w), .. } => *w,
            _ => 0,
        };
        self.configures.lock().unwrap().push(w);
        if w == self.accept_width {
            *self.current_width.lock().unwrap() = w;
            Ok(ConfigureOutcome::Accepted)
        } else {
            Ok(ConfigureOutcome::ReFixate(self.counter.clone()))
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        match &packet {
            PipelinePacket::CapsChanged(c) => {
                if let Caps::RawVideo { width: Dim::Fixed(w), .. } = c {
                    *self.current_width.lock().unwrap() = *w;
                }
            }
            PipelinePacket::DataFrame(_) => {
                let w = *self.current_width.lock().unwrap();
                self.received_data_widths.lock().unwrap().push(w);
            }
            _ => {}
        }
        Box::pin(async { Ok(()) })
    }
}

/// Source that observes `PushOutcome::Reconfigure` mid-stream, drives
/// the trait-level `reconfigure()` hook, and resumes producing under
/// the agreed caps. Hand-shaped to deterministically trigger Reconfigure
/// when paired with [`PickyByWidthSink`].
struct ReconfigurableTestSrc {
    initial: Caps,
    rejected_proposal: Caps, // emit this mid-stream to trigger sink ReFixate
    reconfigure_calls: Mutex<u32>,
}

impl ReconfigurableTestSrc {
    fn calls(&self) -> u32 {
        *self.reconfigure_calls.lock().unwrap()
    }
}

impl SourceLoop for ReconfigurableTestSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.initial.clone()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn reconfigure(&mut self, request: Reconfigure) -> Result<Caps, G2gError> {
        *self.reconfigure_calls.lock().unwrap() += 1;
        match request {
            // For test determinism we accept whatever the sink proposed.
            Reconfigure::Propose(c) => Ok(c),
            Reconfigure::Renegotiate => Err(G2gError::FixationFailed),
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let rejected = self.rejected_proposal.clone();

            // 1. Frame under initial caps , sink configured for initial, accepts.
            out.push(PipelinePacket::DataFrame(make_frame(0)))
                .await?;

            // 2. CapsChanged the sink will reject. Runner fires
            //    Reconfigure(Propose(counter)) on this link's reverse
            //    channel. We don't see the signal yet , push N+1 does.
            out.push(PipelinePacket::CapsChanged(rejected.clone())).await?;

            // 3. Next push observes the Reconfigure. The sink tracks the
            //    current caps via CapsChanged events, so the rejected caps
            //    are what the sink "sees" for this frame. We immediately
            //    handle the reconfigure and emit a fresh CapsChanged the
            //    sink will accept.
            let outcome = out
                .push(PipelinePacket::DataFrame(make_frame(1)))
                .await?;
            let agreed = match outcome {
                PushOutcome::Reconfigure(r) => self.reconfigure(r)?,
                PushOutcome::Accepted => panic!("expected Reconfigure on push N+1"),
            };
            out.push(PipelinePacket::CapsChanged(agreed.clone())).await?;

            // 4. Frame under agreed caps , sink accepts these.
            out.push(PipelinePacket::DataFrame(make_frame(2))).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(3)
        })
    }
}

#[tokio::test]
async fn mid_stream_reconfigure_round_trip() {
    // initial caps width=640: sink accepts (matches accept_width).
    // mid-stream rejected width=1280: sink ReFixates back to initial.
    // source.reconfigure() returns the counter (initial), emits
    // CapsChanged(initial), sink accepts again, pipeline drains.
    let initial = caps_at(640, 480);
    let rejected_mid = caps_at(1280, 720);

    let mut src = ReconfigurableTestSrc {
        initial: initial.clone(),
        rejected_proposal: rejected_mid,
        reconfigure_calls: Mutex::new(0),
    };
    let mut snk = PickyByWidthSink::new(640, initial);
    let clock = ZeroClock;

    // Link capacity 1 forces the source to await capacity between pushes,
    // giving the sink a chance to consume the CapsChanged packet (and
    // fire Reconfigure upstream) before the source pushes its next frame.
    // Larger capacities would let the source race ahead of the sink and
    // miss the signal. Real pipelines with higher capacity will simply
    // observe Reconfigure a few frames later.
    run_simple_pipeline(&mut src, &mut snk, &clock, 1)
        .await
        .expect("round-trip should converge");

    assert_eq!(src.calls(), 1, "source.reconfigure must be invoked once");

    // Sink configure widths: 640 (Phase 2 initial accept), 1280
    // (mid-stream rejected, fires Reconfigure upstream), 640
    // (mid-stream re-accepted after source applied the counter).
    assert_eq!(snk.configure_widths(), vec![640, 1280, 640]);

    // Data frames consumed: 640 throughout, from the sink's perspective.
    // The rejected CapsChanged(1280) is intercepted by the runner before
    // reaching the sink (it triggers a Reconfigure instead), so the
    // sink's tracked width never advances to 1280. The in-flight data
    // frame that arrives between rejection and re-acceptance is observed
    // under the last-accepted width (640).
    assert_eq!(snk.data_widths(), vec![640, 640, 640]);
}

/// Sink that narrows incoming caps against its own supported `Range` in
/// Phase 1 and records the absolute caps it receives in Phase 2, so the
/// test can assert the runner fixated the negotiated range to a single value.
struct IntersectingRecordingSink {
    supported: Caps,
    configured_caps: Mutex<Option<Caps>>,
}

impl IntersectingRecordingSink {
    fn new(supported: Caps) -> Self {
        Self { supported, configured_caps: Mutex::new(None) }
    }

    fn configured_caps(&self) -> Option<Caps> {
        self.configured_caps.lock().unwrap().clone()
    }
}

impl AsyncElement for IntersectingRecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.supported)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        *self.configured_caps.lock().unwrap() = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Source proposes width ∈ [640, 1920]; sink supports width ∈ [1280, 3840].
/// Phase 1 intersects to [1280, 1920]; Phase 2 fixates to the minimum 1280.
/// This is the first test exercising non-trivial `intersect` + `fixate`
/// through the runner.
#[tokio::test]
async fn range_negotiation_intersects_then_fixates_to_minimum() {
    let proposal = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Range { min: 640, max: 1920 },
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    let supported = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Range { min: 1280, max: 3840 },
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let mut src = StaticCapsSrc { proposal };
    let mut snk = IntersectingRecordingSink::new(supported);
    let clock = ZeroClock;

    run_simple_pipeline(&mut src, &mut snk, &clock, 4)
        .await
        .expect("range negotiation must converge");

    let got = snk.configured_caps().expect("sink must be configured");
    assert!(got.is_fixed(), "Phase 2 must hand the sink fully-fixed caps: {got:?}");
    assert_eq!(
        got,
        caps_at(1280, 480),
        "width fixates to the intersected range minimum, height/framerate carried through"
    );
}

#[tokio::test]
async fn caps_changed_triggers_configure_before_next_frame() {
    let mut src = CapsChangingTestSrc {
        initial: caps_at(640, 480),
        refined: caps_at(1920, 1080),
        configured: false,
    };
    let mut snk = OrderedRecordingSink::default();
    let clock = ZeroClock;

    run_simple_pipeline(&mut src, &mut snk, &clock, 4)
        .await
        .expect("pipeline should complete");

    // Expected order:
    //   Configure(640)  — Phase 2 initial fixation
    //   Data(0)         — first frame at initial caps
    //   Configure(1920) — runner cascade on CapsChanged
    //   CapsChanged(1920) — notification packet delivered after configure
    //   Data(1)         — second frame at refined caps
    //   Eos
    let events = snk.events();
    assert_eq!(
        events,
        vec![
            Event::Configure { width: 640 },
            Event::Data { seq: 0 },
            Event::Configure { width: 1920 },
            Event::CapsChanged { width: 1920 },
            Event::Data { seq: 1 },
            Event::Eos,
        ],
        "unexpected event order: {events:?}"
    );
}
