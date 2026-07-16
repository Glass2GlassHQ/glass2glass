//! Runtime fault recovery for the static heap-free MCU path: a supervisor that
//! wraps a `source -> sink` chain and turns a returned fault
//! ([`G2gError`](crate::error::G2gError)) into a *bounded, deterministic*
//! recovery action instead of aborting the pipeline.
//!
//! The static runners ([`run_source_sink`](crate::staticelem::run_source_sink),
//! [`step_source_sink`](crate::staticelem::step_source_sink)) propagate a fault
//! straight out: the first `PoolExhausted` / `Hardware(Peripheral)` / overrun
//! ends the pipeline. That is correct for a host tool, but the MCU / safety
//! market needs the opposite default: a transient peripheral glitch should be
//! retried or degraded around, a persistent one should re-initialize the stages,
//! and only an unrecoverable fault should stop, at which point an unpetted
//! hardware watchdog resets the chip. This module supplies that policy layer, in
//! the no-alloc subset, so it links on a target with no allocator.
//!
//! The pieces:
//! - [`FaultPolicy`] classifies each fault into a [`Recovery`] action. The
//!   supplied [`RetryThenReset`] and [`SkipBounded`] cover the two canonical
//!   safety patterns (recover-in-place vs. degrade-and-continue); both are
//!   bounded, so a persistent fault always escalates in finite steps.
//! - [`Recover`] is the per-stage re-initialization seam (re-arm DMA, flush a
//!   partial packet, reset a decoder). Its default is a no-op, so a stateless
//!   stage opts in with `impl Recover for Foo {}`.
//! - [`Watchdog`] is petted on every frame of real forward progress; when the
//!   supervisor stops (a wedged or escalated pipeline) the watchdog is no longer
//!   fed, so a hardware watchdog fires and resets the MCU.
//! - [`SupervisorReport`] is the fault accounting (frames / faults / retries /
//!   resets / skips / escalation) the safety case wants for traceability.
//!
//! Everything is bounded by construction: [`step_supervised`] resolves each frame
//! in at most [`MAX_ATTEMPTS`] internal iterations regardless of the policy, so a
//! buggy policy can never spin the supervisor forever.

use crate::error::G2gError;
use crate::staticelem::{drive_ready, Chain, SinkChain, SourceChain, StaticSink, StaticSource};

/// What the supervisor does about a fault, decided by a [`FaultPolicy`].
///
/// The four actions are the deterministic MCU fault-handling vocabulary:
/// try again, drop and keep the cadence, re-initialize, or give up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    /// Re-drive the step immediately (a transient fault worth another attempt:
    /// a momentarily exhausted pool, a one-off capture timeout). Because the
    /// step pulls the *next* frame afresh, this suits source-side and transient
    /// faults; the frame that faulted is not buffered for replay (that would
    /// break the zero-copy single-frame-in-flight model).
    Retry,
    /// Drop this frame and return control (degraded mode): the stream keeps
    /// flowing at cadence past a corrupt or late frame. The consecutive-fault
    /// count is *not* cleared, so a source that only ever skips still escalates
    /// in finite steps and never pets the watchdog into thinking it is healthy.
    Skip,
    /// Re-initialize both stages via their [`Recover`] hook, then return control.
    /// For a fault a fresh peripheral state clears (re-arm the DMA, reset the
    /// codec). A failing recover escalates.
    Reset,
    /// Give up: the fault is structural or the retry/reset budget is spent. The
    /// supervisor returns it and does not pet the watchdog, so on hardware the
    /// watchdog fires; a caller can also trigger a system reset explicitly.
    Escalate,
}

/// Classifies a fault into a [`Recovery`] action, given the fault and how many
/// consecutive faults have occurred since the last frame of real progress. An
/// implementation must be *bounded*: for a persistently faulting stage it must
/// eventually return [`Recovery::Escalate`], so the supervisor cannot livelock.
pub trait FaultPolicy {
    /// Decide what to do about `err`, the `consecutive`-th fault in a row
    /// (1 on the first fault after a good frame, climbing until one succeeds).
    fn classify(&mut self, err: &G2gError, consecutive: u32) -> Recovery;
}

/// Bounded recover-in-place policy: retry a fault up to `max_retries` times, then
/// re-initialize the stages up to `max_resets` times, then escalate. Structural
/// faults (a caps / configuration / copy-budget violation) never retry, because
/// re-running the same negotiation cannot succeed; they escalate at once. This is
/// the default for a capture pipeline, where re-arming the peripheral is the
/// natural response to a bus glitch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryThenReset {
    /// Consecutive faults answered with [`Recovery::Retry`] before escalating to
    /// a reset.
    pub max_retries: u32,
    /// Consecutive faults (after the retry budget) answered with
    /// [`Recovery::Reset`] before giving up.
    pub max_resets: u32,
}

impl RetryThenReset {
    /// A policy that retries `max_retries` times, then resets `max_resets` times,
    /// then escalates.
    pub const fn new(max_retries: u32, max_resets: u32) -> Self {
        Self { max_retries, max_resets }
    }
}

impl Default for RetryThenReset {
    /// Two retries, then one reset, then escalate: a transient glitch is retried,
    /// a sticky one triggers a single re-init, a persistent one stops.
    fn default() -> Self {
        Self::new(2, 1)
    }
}

/// A fault the [`RetryThenReset`] ladder treats as structural (never transient):
/// re-running the exact same operation cannot clear it, so it escalates at once.
fn is_structural(err: &G2gError) -> bool {
    matches!(
        err,
        G2gError::CapsMismatch
            | G2gError::NotConfigured
            | G2gError::FixationFailed
            | G2gError::UnsupportedDomain
            | G2gError::AllocationConflict
            | G2gError::CopyBudget
    )
}

impl FaultPolicy for RetryThenReset {
    fn classify(&mut self, err: &G2gError, consecutive: u32) -> Recovery {
        if is_structural(err) {
            return Recovery::Escalate;
        }
        // saturating so a huge consecutive count cannot wrap the threshold and
        // wrongly re-enter the retry band.
        let reset_ceiling = self.max_retries.saturating_add(self.max_resets);
        if consecutive <= self.max_retries {
            Recovery::Retry
        } else if consecutive <= reset_ceiling {
            Recovery::Reset
        } else {
            Recovery::Escalate
        }
    }
}

/// Bounded degrade-and-continue policy: drop up to `max_skips` consecutive
/// faulting frames (keeping the output cadence), then escalate. The default for a
/// display / telemetry pipeline where a dropped frame is the safe degraded state
/// and re-arming per glitch is not worth the stall. Structural faults escalate at
/// once, as in [`RetryThenReset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipBounded {
    /// Consecutive faulting frames dropped before escalating.
    pub max_skips: u32,
}

impl SkipBounded {
    /// A policy that skips up to `max_skips` consecutive faults, then escalates.
    pub const fn new(max_skips: u32) -> Self {
        Self { max_skips }
    }
}

impl Default for SkipBounded {
    fn default() -> Self {
        Self::new(4)
    }
}

impl FaultPolicy for SkipBounded {
    fn classify(&mut self, err: &G2gError, consecutive: u32) -> Recovery {
        if is_structural(err) || consecutive > self.max_skips {
            Recovery::Escalate
        } else {
            Recovery::Skip
        }
    }
}

/// Per-stage re-initialization seam, called on a [`Recovery::Reset`]: bring the
/// stage back to a known-good state so the next frame starts clean (re-arm the
/// capture DMA, flush a half-written packet, reset a decoder's reference state).
///
/// The default is a no-op, so a stateless stage opts in with `impl Recover for
/// Foo {}` and a stage with peripheral state overrides `recover`. A supervised
/// pipeline therefore *declares* each stage's recovery behavior, which is the
/// traceability a safety case needs. A `recover` that itself fails escalates the
/// supervisor.
///
/// `#[allow(async_fn_in_trait)]` for the same reason as the rest of the static
/// element model: a single-executor MCU path, no `Send` needed, no boxing.
#[allow(async_fn_in_trait)]
pub trait Recover {
    /// Re-initialize after a fault. The default is a no-op (a stateless stage).
    async fn recover(&mut self) -> Result<(), G2gError> {
        Ok(())
    }
}

/// A watchdog the supervisor pets on every frame of real forward progress. On an
/// MCU this refreshes a hardware watchdog timer (STM32 IWDG, an RTOS software
/// watchdog); when the supervisor stops petting, a wedged or escalated pipeline,
/// the timer expires and resets the chip. This is the backstop for a pipeline
/// that neither advances nor escalates (e.g. a stage that hangs an interrupt):
/// petting only on `Advanced` means "no real frame for too long" trips it.
pub trait Watchdog {
    /// Refresh the watchdog; called once per delivered frame.
    fn pet(&mut self);
}

/// A no-op watchdog for a pipeline that does not use one (host tests, a target
/// whose reset is handled elsewhere).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoWatchdog;

impl Watchdog for NoWatchdog {
    fn pet(&mut self) {}
}

// So a caller keeps its watchdog to inspect after the run (the runners take
// ownership), a `&mut W` is itself a watchdog.
impl<W: Watchdog> Watchdog for &mut W {
    fn pet(&mut self) {
        (**self).pet();
    }
}

/// Running fault accounting across supervised steps: the evidence a safety case
/// wants (how many faults occurred, how they were handled, whether the pipeline
/// escalated). A caller keeps one across a whole run (or a C superloop keeps one
/// across `step_supervised` calls) and inspects it at the end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisorReport {
    /// Frames of real forward progress delivered to the sink.
    pub frames: u32,
    /// Total faults observed (across all handling actions).
    pub faults: u32,
    /// Faults answered by re-driving the step.
    pub retries: u32,
    /// Faults answered by re-initializing the stages.
    pub resets: u32,
    /// Faults answered by dropping the frame (degraded mode).
    pub skips: u32,
    /// Faults in a row since the last delivered frame (the value the policy sees).
    pub consecutive_faults: u32,
    /// Whether the supervisor has escalated (given up on a fault).
    pub escalated: bool,
    /// The most recent fault observed, if any.
    pub last_error: Option<G2gError>,
}

impl SupervisorReport {
    /// A fresh report with all counters zero.
    pub const fn new() -> Self {
        Self {
            frames: 0,
            faults: 0,
            retries: 0,
            resets: 0,
            skips: 0,
            consecutive_faults: 0,
            escalated: false,
            last_error: None,
        }
    }
}

impl Default for SupervisorReport {
    fn default() -> Self {
        Self::new()
    }
}

/// The outcome of one [`step_supervised`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Supervised {
    /// A frame flowed to the sink; the watchdog was petted.
    Advanced,
    /// A faulting frame was dropped (degraded mode); call again for the next.
    Skipped,
    /// The stages were re-initialized after a fault; call again.
    Recovered,
    /// The source reached end of stream; stop.
    Eos,
    /// A stage suspended (`Poll::Pending`). Like [`Step::Pending`], the supervisor
    /// targets synchronous static stages; a genuinely suspending pipeline belongs
    /// on a real executor. Reported, never silently looped.
    ///
    /// [`Step::Pending`]: crate::staticelem::Step
    Pending,
    /// The fault could not be recovered within the policy's bounds. The watchdog
    /// was not petted; on hardware it now fires. The caller should stop (or
    /// trigger a system reset).
    Escalated(G2gError),
}

/// The hard upper bound on internal iterations [`step_supervised`] performs to
/// resolve one frame, regardless of the [`FaultPolicy`]. A correct bounded policy
/// escalates far below this; the cap is the belt-and-suspenders guarantee that a
/// *buggy* policy (one that never escalates) still cannot hang the supervisor.
pub const MAX_ATTEMPTS: u32 = 64;

/// One attempt at pulling and delivering a frame: `Ok(true)` a frame flowed,
/// `Ok(false)` end of stream, `Err` a fault. Named `async fn` so it monomorphizes
/// like the [`step_source_sink`](crate::staticelem::step_source_sink) body.
async fn attempt<S, K>(src: &mut S, sink: &mut K) -> Result<bool, G2gError>
where
    S: StaticSource,
    K: StaticSink,
{
    match src.next().await? {
        Some(frame) => {
            sink.consume(frame).await?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Re-initialize both stages (source first, then sink) for a [`Recovery::Reset`].
async fn recover_both<S, K>(src: &mut S, sink: &mut K) -> Result<(), G2gError>
where
    S: Recover,
    K: Recover,
{
    src.recover().await?;
    sink.recover().await?;
    Ok(())
}

/// Run one frame through a supervised `source -> sink` chain, applying `policy` to
/// any fault and petting `watchdog` on success, accumulating into `report`. The
/// caller owns the loop (a C superloop over the FFI seam, an RTOS task), so it can
/// interleave other work and feed its own watchdog between frames; use
/// [`run_supervised`] for the run-to-completion case.
///
/// Bounded: resolves in at most [`MAX_ATTEMPTS`] internal iterations whatever the
/// policy does. Compose a transform tail into the sink with
/// [`SinkChain`](crate::staticelem::SinkChain) (and a head with
/// [`SourceChain`](crate::staticelem::SourceChain)) so this steps any linear
/// graph; both combinators forward [`Recover`] to their parts.
pub fn step_supervised<S, K, P, W>(
    src: &mut S,
    sink: &mut K,
    policy: &mut P,
    watchdog: &mut W,
    report: &mut SupervisorReport,
) -> Supervised
where
    S: StaticSource + Recover,
    K: StaticSink + Recover,
    P: FaultPolicy,
    W: Watchdog,
{
    let mut iterations = 0;
    while iterations < MAX_ATTEMPTS {
        iterations += 1;
        match drive_ready(attempt(src, sink)) {
            Some(Ok(true)) => {
                report.frames = report.frames.saturating_add(1);
                report.consecutive_faults = 0;
                watchdog.pet();
                return Supervised::Advanced;
            }
            Some(Ok(false)) => return Supervised::Eos,
            Some(Err(err)) => {
                report.faults = report.faults.saturating_add(1);
                report.consecutive_faults = report.consecutive_faults.saturating_add(1);
                report.last_error = Some(err.clone());
                match policy.classify(&err, report.consecutive_faults) {
                    Recovery::Retry => {
                        report.retries = report.retries.saturating_add(1);
                        // loop: re-drive the step (bounded by MAX_ATTEMPTS).
                    }
                    Recovery::Skip => {
                        report.skips = report.skips.saturating_add(1);
                        return Supervised::Skipped;
                    }
                    Recovery::Reset => {
                        report.resets = report.resets.saturating_add(1);
                        match drive_ready(recover_both(src, sink)) {
                            Some(Ok(())) => return Supervised::Recovered,
                            Some(Err(rerr)) => {
                                report.escalated = true;
                                return Supervised::Escalated(rerr);
                            }
                            // recover_both suspended: a re-init that awaits belongs
                            // on a real executor, escalate rather than spin.
                            None => {
                                report.escalated = true;
                                return Supervised::Escalated(err);
                            }
                        }
                    }
                    Recovery::Escalate => {
                        report.escalated = true;
                        return Supervised::Escalated(err);
                    }
                }
            }
            None => return Supervised::Pending,
        }
    }
    // Hard cap reached (a policy that never escalated): force a bounded stop.
    report.escalated = true;
    Supervised::Escalated(report.last_error.clone().unwrap_or(G2gError::Shutdown))
}

/// How a supervised run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// The source reached end of stream; the pipeline completed.
    Completed,
    /// A fault could not be recovered within the policy's bounds; the pipeline
    /// stopped. The watchdog was not petted since the last delivered frame, so on
    /// hardware it now resets the chip.
    Escalated(G2gError),
    /// A stage suspended. The synchronous supervisor cannot drive it; run this
    /// pipeline on a real executor instead.
    Suspended,
}

/// Drive a supervised `source -> sink` chain to end of stream (or escalation),
/// returning the fault accounting and the terminal outcome. The run-to-completion
/// analog of [`step_supervised`]; a caller that must interleave other work uses
/// the step form instead.
pub fn run_supervised<S, K, P, W>(
    mut src: S,
    mut sink: K,
    mut policy: P,
    mut watchdog: W,
) -> (SupervisorReport, RunOutcome)
where
    S: StaticSource + Recover,
    K: StaticSink + Recover,
    P: FaultPolicy,
    W: Watchdog,
{
    let mut report = SupervisorReport::new();
    loop {
        match step_supervised(&mut src, &mut sink, &mut policy, &mut watchdog, &mut report) {
            Supervised::Advanced | Supervised::Skipped | Supervised::Recovered => {}
            Supervised::Eos => return (report, RunOutcome::Completed),
            Supervised::Escalated(err) => return (report, RunOutcome::Escalated(err)),
            Supervised::Pending => return (report, RunOutcome::Suspended),
        }
    }
}

// Recover forwarding for the static-chain combinators and `&mut`, so a supervised
// sink built as `SinkChain(transform, sink)` (or a source as `SourceChain`)
// recovers all of its parts. Each recurses; a bare stage supplies its own (the
// default no-op unless it has peripheral state).

impl<T: Recover> Recover for &mut T {
    async fn recover(&mut self) -> Result<(), G2gError> {
        (**self).recover().await
    }
}

impl<A: Recover, B: Recover> Recover for Chain<A, B> {
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.0.recover().await?;
        self.1.recover().await
    }
}

impl<S: Recover, T: Recover> Recover for SourceChain<S, T> {
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.0.recover().await?;
        self.1.recover().await
    }
}

impl<T: Recover, K: Recover> Recover for SinkChain<T, K> {
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.0.recover().await?;
        self.1.recover().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};

    static BYTE: [u8; 1] = [7];

    fn one_frame(seq: u64) -> Frame {
        // SAFETY: BYTE is 'static and never mutated; the lent slice covers its one
        // valid byte and needs no reclamation (free = None).
        let slice = unsafe {
            SystemSlice::from_foreign(BYTE.as_ptr(), 1, None, core::ptr::null_mut())
        };
        Frame::new(
            MemoryDomain::System(slice),
            FrameTiming { pts_ns: seq, ..FrameTiming::default() },
            seq,
        )
    }

    /// A source that faults on a configurable set of (attempt-)indices, counting
    /// how many times it was re-initialized. `fault_first_n_of_each` makes the
    /// fault transient: the k-th distinct frame faults on its first attempt and
    /// succeeds on the retry.
    struct FaultSource {
        emitted: u32,
        limit: u32,
        // faults still owed on the current frame before it will succeed.
        fault_countdown: u32,
        faults_per_frame: u32,
        recovers: u32,
        // once true, every attempt faults (the permanent-fault case).
        permanent: bool,
    }
    impl FaultSource {
        fn transient(limit: u32, faults_per_frame: u32) -> Self {
            Self {
                emitted: 0,
                limit,
                fault_countdown: faults_per_frame,
                faults_per_frame,
                recovers: 0,
                permanent: false,
            }
        }
        fn permanent() -> Self {
            Self {
                emitted: 0,
                limit: 1,
                fault_countdown: 1,
                faults_per_frame: 1,
                recovers: 0,
                permanent: true,
            }
        }
    }
    impl StaticSource for FaultSource {
        async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
            if !self.permanent && self.emitted >= self.limit {
                return Ok(None);
            }
            if self.permanent {
                return Err(G2gError::Hardware(crate::error::HardwareError::Peripheral));
            }
            if self.fault_countdown > 0 {
                self.fault_countdown -= 1;
                return Err(G2gError::PoolExhausted);
            }
            let seq = self.emitted as u64;
            self.emitted += 1;
            self.fault_countdown = self.faults_per_frame; // arm the next frame's faults
            Ok(Some(one_frame(seq)))
        }
    }
    impl Recover for FaultSource {
        async fn recover(&mut self) -> Result<(), G2gError> {
            self.recovers += 1;
            Ok(())
        }
    }

    struct CountSink {
        n: u32,
    }
    impl StaticSink for CountSink {
        async fn consume(&mut self, _frame: Frame) -> Result<(), G2gError> {
            self.n += 1;
            Ok(())
        }
    }
    impl Recover for CountSink {}

    struct CountWatchdog {
        pets: u32,
    }
    impl Watchdog for CountWatchdog {
        fn pet(&mut self) {
            self.pets += 1;
        }
    }

    #[test]
    fn transient_faults_are_retried_and_every_frame_still_arrives() {
        // Each of 5 frames faults once (PoolExhausted) then succeeds on retry.
        let src = FaultSource::transient(5, 1);
        let mut sink = CountSink { n: 0 };
        let mut wd = CountWatchdog { pets: 0 };
        let (report, outcome) =
            run_supervised(src, &mut sink, RetryThenReset::default(), &mut wd);
        assert_eq!(outcome, RunOutcome::Completed, "recovered to EOS");
        assert_eq!(sink.n, 5, "all frames delivered despite the transient faults");
        assert_eq!(report.frames, 5);
        assert_eq!(report.faults, 5, "one fault per frame");
        assert_eq!(report.retries, 5, "each answered by a retry");
        assert_eq!(report.resets, 0);
        assert!(!report.escalated);
        assert_eq!(wd.pets, 5, "watchdog petted once per delivered frame");
    }

    #[test]
    fn sticky_fault_triggers_a_reset_then_recovers() {
        // Two faults per frame: exceeds the 2-retry budget, so the 3rd consecutive
        // fault triggers a reset; the frame then succeeds. faults_per_frame=2 means
        // frame k faults twice then emits.
        let src = FaultSource::transient(3, 2);
        let mut sink = CountSink { n: 0 };
        let (report, outcome) =
            run_supervised(src, &mut sink, RetryThenReset::new(2, 1), NoWatchdog);
        assert_eq!(outcome, RunOutcome::Completed);
        assert_eq!(sink.n, 3, "all frames eventually delivered");
        // Per frame: fault, fault -> both under the 2-retry budget (consecutive 1,2)
        // so 2 retries, then the frame succeeds (consecutive resets to 0). No reset
        // is reached because 2 faults <= max_retries.
        assert_eq!(report.retries, 6, "two retries per frame, three frames");
        assert_eq!(report.resets, 0, "two faults stays within the retry budget");
        assert!(!report.escalated);
    }

    #[test]
    fn exceeding_the_retry_budget_escalates_to_reset() {
        // Three faults per frame with a 2-retry / 1-reset budget: consecutive
        // 1,2 -> retry, 3 -> reset (clears the peripheral in this mock), 4th
        // attempt succeeds.
        let src = FaultSource::transient(2, 3);
        let mut sink = CountSink { n: 0 };
        let (report, outcome) =
            run_supervised(src, &mut sink, RetryThenReset::new(2, 1), NoWatchdog);
        assert_eq!(outcome, RunOutcome::Completed);
        assert_eq!(sink.n, 2);
        assert_eq!(report.resets, 2, "one reset per frame (2 frames)");
        assert!(!report.escalated);
    }

    #[test]
    fn permanent_fault_escalates_within_bounds_and_stops_petting() {
        let src = FaultSource::permanent();
        let mut sink = CountSink { n: 0 };
        let mut wd = CountWatchdog { pets: 0 };
        let (report, outcome) =
            run_supervised(src, &mut sink, RetryThenReset::new(2, 1), &mut wd);
        match outcome {
            RunOutcome::Escalated(G2gError::Hardware(_)) => {}
            other => panic!("expected escalation on a permanent fault, got {other:?}"),
        }
        assert_eq!(sink.n, 0, "no frame ever delivered");
        assert!(report.escalated);
        // 2 retries + 1 reset then escalate on the 4th consecutive fault.
        assert_eq!(report.faults, 4);
        assert_eq!(report.retries, 2);
        assert_eq!(report.resets, 1);
        assert_eq!(wd.pets, 0, "watchdog never petted -> hardware reset fires");
    }

    #[test]
    fn structural_fault_escalates_immediately_without_retrying() {
        // A CapsMismatch is structural: no retry, straight to escalation.
        struct CapsFault;
        impl StaticSource for CapsFault {
            async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
                Err(G2gError::CapsMismatch)
            }
        }
        impl Recover for CapsFault {}
        let mut sink = CountSink { n: 0 };
        let (report, outcome) =
            run_supervised(CapsFault, &mut sink, RetryThenReset::default(), NoWatchdog);
        assert_eq!(outcome, RunOutcome::Escalated(G2gError::CapsMismatch));
        assert_eq!(report.faults, 1, "escalated on the first fault, no retries");
        assert_eq!(report.retries, 0);
        assert_eq!(report.resets, 0);
    }

    #[test]
    fn skip_policy_degrades_past_faults_and_keeps_the_good_frames() {
        // Every other frame faults; SkipBounded drops the faulters and delivers the
        // rest, staying at cadence, until EOS.
        struct AlternatingFault {
            emitted: u32,
            attempts: u32,
            limit: u32,
        }
        impl StaticSource for AlternatingFault {
            async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
                if self.emitted >= self.limit {
                    return Ok(None);
                }
                self.attempts += 1;
                // Fault once on every 3rd attempt.
                if self.attempts % 3 == 0 {
                    return Err(G2gError::Hardware(crate::error::HardwareError::Peripheral));
                }
                let seq = self.emitted as u64;
                self.emitted += 1;
                Ok(Some(one_frame(seq)))
            }
        }
        impl Recover for AlternatingFault {}
        let src = AlternatingFault { emitted: 0, attempts: 0, limit: 6 };
        let mut sink = CountSink { n: 0 };
        let (report, outcome) =
            run_supervised(src, &mut sink, SkipBounded::new(2), NoWatchdog);
        assert_eq!(outcome, RunOutcome::Completed, "degraded past the faults to EOS");
        assert_eq!(sink.n, 6, "every good frame delivered");
        assert!(report.skips > 0, "faulting frames were skipped, not retried");
        assert_eq!(report.retries, 0);
        assert!(!report.escalated);
    }

    #[test]
    fn a_never_escalating_policy_still_stops_at_the_hard_cap() {
        // A pathological policy that always retries: the MAX_ATTEMPTS belt forces a
        // bounded stop so the supervisor cannot hang.
        struct AlwaysRetry;
        impl FaultPolicy for AlwaysRetry {
            fn classify(&mut self, _err: &G2gError, _consecutive: u32) -> Recovery {
                Recovery::Retry
            }
        }
        let src = FaultSource::permanent();
        let mut sink = CountSink { n: 0 };
        let mut report = SupervisorReport::new();
        let out = step_supervised(
            &mut { src },
            &mut sink,
            &mut AlwaysRetry,
            &mut NoWatchdog,
            &mut report,
        );
        assert!(matches!(out, Supervised::Escalated(_)), "hard cap forced a stop");
        assert!(report.escalated);
        assert_eq!(report.faults, MAX_ATTEMPTS, "stopped exactly at the cap");
    }
}
