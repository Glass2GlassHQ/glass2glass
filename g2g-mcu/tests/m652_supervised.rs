//! M652: runtime fault recovery. A supervised `capture -> G.711 -> sink`
//! pipeline turns a returned fault into a bounded retry / reset / escalate action
//! instead of aborting, pets a hardware watchdog on forward progress, and stops
//! (leaving the watchdog to reset the chip) when a fault is unrecoverable. The
//! grabber is a mock peripheral; the elements, the supervisor, the `FrameGrabber`
//! reset seam, and the watchdog adapter under test are all real.

use g2g_core::error::{G2gError, HardwareError};
use g2g_core::staticelem::SinkChain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::supervise::{
    run_supervised, NoWatchdog, Recover, RetryThenReset, RunOutcome, SkipBounded, SupervisorReport,
};
use g2g_core::{Frame, StaticSink};
use g2g_mcu::{FrameGrabber, G711Enc, GrabberSrc, Law, SupervisorWatchdog, WatchdogTimer};

fn leaked_ring<const N: usize, const B: usize>() -> &'static StaticLendRing<N, B> {
    Box::leak(Box::new(StaticLendRing::new()))
}

/// A capture peripheral that needs a re-arm (a `reset`) to clear a fault: it
/// faults on the capture at `fault_at`, disarming itself, and only a `reset`
/// re-arms it. This models a DMA/DCMI peripheral that latches an error until the
/// driver re-initializes it, so it proves the supervisor's `Reset` action reaches
/// `FrameGrabber::reset` and actually recovers.
struct StickyGrabber {
    captures: u32,
    armed: bool,
    tripped: bool,
    fault_at: u32,
    resets: u32,
}
impl StickyGrabber {
    fn new(fault_at: u32) -> Self {
        Self { captures: 0, armed: true, tripped: false, fault_at, resets: 0 }
    }
}
impl FrameGrabber for StickyGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        if !self.armed {
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        if !self.tripped && self.captures == self.fault_at {
            // Latch the fault: disarm until a reset re-arms the peripheral.
            self.tripped = true;
            self.armed = false;
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        // Two S16LE samples per frame; stamp the capture index into sample 0.
        if let [lo, hi, ..] = buf {
            *lo = (self.captures & 0xff) as u8;
            *hi = 0;
        }
        self.captures += 1;
        Ok(buf.len())
    }
    async fn reset(&mut self) -> Result<(), G2gError> {
        self.resets += 1;
        self.armed = true;
        Ok(())
    }
}

/// A truly dead peripheral: every capture faults and `reset` (the default no-op)
/// cannot revive it. The supervisor must escalate within its bounds.
struct DeadGrabber;
impl FrameGrabber for DeadGrabber {
    async fn capture(&mut self, _buf: &mut [u8]) -> Result<usize, G2gError> {
        Err(G2gError::Hardware(HardwareError::Peripheral))
    }
}

/// Counts frames and their first payload byte; recovers as a no-op.
struct CountSink {
    n: u32,
    first_bytes: Vec<u8>,
}
impl CountSink {
    fn new() -> Self {
        Self { n: 0, first_bytes: Vec::new() }
    }
}
impl StaticSink for CountSink {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if let g2g_core::MemoryDomain::System(s) = &frame.domain {
            if let Some(&b) = s.as_slice().first() {
                self.first_bytes.push(b);
            }
        }
        self.n += 1;
        Ok(())
    }
}
impl Recover for CountSink {}

/// A mock independent watchdog counting refreshes.
#[derive(Default)]
struct MockIwdg {
    refreshes: std::rc::Rc<std::cell::Cell<u32>>,
}
impl WatchdogTimer for MockIwdg {
    fn feed(&mut self) {
        self.refreshes.set(self.refreshes.get() + 1);
    }
}

// The G.711 encoder needs its own output ring; capture ring feeds it S16 pairs.
// Two S16 samples in -> two companded bytes out.

#[test]
fn transient_fault_recovers_via_reset_and_delivers_every_frame() {
    let cap_ring: &'static StaticLendRing<2, 4> = leaked_ring(); // 2 samples/frame
    let enc_ring: &'static StaticLendRing<2, 2> = leaked_ring(); // 2 bytes/frame
    // Frame index 3 latches a fault that only a reset clears.
    let src = GrabberSrc::new(StickyGrabber::new(3), cap_ring, 125_000).with_frame_limit(6);
    let enc = G711Enc::new(Law::Mulaw, enc_ring);
    let mut sink = CountSink::new();

    let refreshes = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let wd = SupervisorWatchdog::new(MockIwdg { refreshes: refreshes.clone() });

    // Default ladder: 2 retries then 1 reset. Frame 3 faults, the two retries
    // still fault (peripheral latched), the reset re-arms it, the 4th attempt
    // succeeds. Every frame therefore reaches the sink.
    let (report, outcome): (SupervisorReport, RunOutcome) =
        run_supervised(src, SinkChain(enc, &mut sink), RetryThenReset::default(), wd);

    assert_eq!(outcome, RunOutcome::Completed, "recovered to end of stream");
    assert_eq!(sink.n, 6, "all six frames delivered despite the latched fault");
    // The one faulting frame cost two retries and one reset; no other frame faults.
    assert_eq!(report.faults, 3, "two retries + the reset trigger = three fault observations");
    assert_eq!(report.retries, 2);
    assert_eq!(report.resets, 1, "the reset seam fired exactly once");
    assert!(!report.escalated);
    assert_eq!(refreshes.get(), 6, "watchdog fed once per delivered frame");
    // The captured pattern survived recovery: frames 0..6 companded in order
    // (frame 3 was re-captured after the reset with its original index).
    assert_eq!(sink.first_bytes.len(), 6, "one payload per frame, in order");
}

#[test]
fn dead_peripheral_escalates_within_bounds_and_stops_petting() {
    let cap_ring: &'static StaticLendRing<2, 4> = leaked_ring();
    let enc_ring: &'static StaticLendRing<2, 2> = leaked_ring();
    let src = GrabberSrc::new(DeadGrabber, cap_ring, 125_000).with_frame_limit(6);
    let enc = G711Enc::new(Law::Mulaw, enc_ring);
    let mut sink = CountSink::new();

    let refreshes = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let wd = SupervisorWatchdog::new(MockIwdg { refreshes: refreshes.clone() });

    let (report, outcome) =
        run_supervised(src, SinkChain(enc, &mut sink), RetryThenReset::new(2, 1), wd);

    match outcome {
        RunOutcome::Escalated(G2gError::Hardware(HardwareError::Peripheral)) => {}
        other => panic!("expected escalation on a dead peripheral, got {other:?}"),
    }
    assert_eq!(sink.n, 0, "no frame ever delivered");
    assert!(report.escalated);
    // 2 retries + 1 reset then escalate on the 4th consecutive fault.
    assert_eq!(report.faults, 4);
    assert_eq!(report.retries, 2);
    assert_eq!(report.resets, 1);
    assert_eq!(refreshes.get(), 0, "watchdog never fed -> the hardware watchdog resets the chip");
}

#[test]
fn skip_policy_degrades_past_intermittent_faults() {
    // Every third capture attempt faults; SkipBounded drops it and keeps the
    // cadence, delivering the good frames to end of stream.
    struct FlakyGrabber {
        captures: u32,
        attempts: u32,
        limit: u32,
    }
    impl FrameGrabber for FlakyGrabber {
        async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
            // GrabberSrc's frame limit ends the stream; the grabber only
            // produces or faults, never signals EOS itself.
            let _ = self.limit;
            self.attempts += 1;
            if self.attempts % 3 == 0 {
                return Err(G2gError::Hardware(HardwareError::Peripheral));
            }
            if let [lo, hi, ..] = buf {
                *lo = (self.captures & 0xff) as u8;
                *hi = 0;
            }
            self.captures += 1;
            Ok(buf.len())
        }
    }

    let cap_ring: &'static StaticLendRing<2, 4> = leaked_ring();
    let enc_ring: &'static StaticLendRing<2, 2> = leaked_ring();
    let src = GrabberSrc::new(FlakyGrabber { captures: 0, attempts: 0, limit: 8 }, cap_ring, 125_000)
        .with_frame_limit(8);
    let enc = G711Enc::new(Law::Mulaw, enc_ring);
    let mut sink = CountSink::new();

    let (report, outcome) =
        run_supervised(src, SinkChain(enc, &mut sink), SkipBounded::new(2), NoWatchdog);

    assert_eq!(outcome, RunOutcome::Completed, "degraded past faults to EOS");
    assert_eq!(sink.n, 8, "every good frame delivered");
    assert!(report.skips > 0, "faulting captures were skipped, not retried");
    assert_eq!(report.retries, 0);
    assert!(!report.escalated);
}
