//! Runtime fault-recovery proof (M652): the supervisor on the emulated
//! Cortex-M4. A `capture -> G.711 -> checksum` pipeline is driven by
//! `run_supervised`, which turns a returned peripheral fault into a bounded
//! retry / reset / escalate action instead of aborting, and pets a hardware
//! watchdog (through the `g2g-mcu` `WatchdogTimer` seam) on every frame of real
//! progress.
//!
//! Two scenarios run on-target:
//!
//! 1. Recover: the capture peripheral latches a fault mid-stream that only a
//!    `reset` clears. The supervisor retries, then resets (re-arming the mock
//!    peripheral via `FrameGrabber::reset`), then continues, so every frame is
//!    still delivered and the wire checksum equals a clean synchronous reference
//!    run. The watchdog is fed exactly once per delivered frame.
//!
//! 2. Escalate: a dead peripheral faults forever. The supervisor exhausts its
//!    bounded ladder (2 retries + 1 reset) and escalates in finite steps rather
//!    than hanging; the watchdog is never fed, which on real silicon is what lets
//!    the hardware watchdog reset the chip.
//!
//! `tools/qemu-check.sh` boots this and asserts the banner + semihosting exit,
//! upgrading the host-side supervisor tests to "runs on the Cortex-M ISA".

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};

use g2g_core::error::{G2gError, HardwareError};
use g2g_core::staticpool::StaticLendRing;
use g2g_core::supervise::{run_supervised, Recover, RetryThenReset, RunOutcome};
use g2g_core::{drive_ready, run_source_transform_sink, Frame, MemoryDomain, SinkChain, StaticSink};
use g2g_mcu::{FrameGrabber, G711Enc, GrabberSrc, Law, SupervisorWatchdog, WatchdogTimer};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// 16 mono S16LE samples per frame: 32 capture bytes, 16 mu-law bytes.
const SAMPLES: usize = 16;
const CAP_BYTES: usize = SAMPLES * 2;
const ULAW_BYTES: usize = SAMPLES;
/// Frames to capture and process.
const TARGET: u32 = 64;
const FRAME_NS: u64 = 1_000_000;
/// The capture seq that latches a fault only a reset clears (mid-stream).
const FAULT_AT: u32 = 30;

/// Deterministic frame content from the capture sequence (a signed ramp sweeping
/// several G.711 segments). Panic-free: slice pattern, no indexing.
fn fill_frame(buf: &mut [u8], seq: u32) {
    for (i, pair) in buf.chunks_exact_mut(2).enumerate() {
        let sample = (((seq.wrapping_add(i as u32)) & 0xff) as i32 - 128) * 128;
        let bytes = (sample as i16).to_le_bytes();
        if let [lo, hi] = pair {
            *lo = bytes[0];
            *hi = bytes[1];
        }
    }
}

/// A checksumming sink: sums every encoded byte and counts frames.
struct SumSink {
    sum: u64,
    count: u32,
}
impl StaticSink for SumSink {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if let MemoryDomain::System(s) = &frame.domain {
            for &b in s.as_slice() {
                self.sum = self.sum.wrapping_add(b as u64);
            }
        }
        self.count = self.count.wrapping_add(1);
        Ok(())
    }
}
impl Recover for SumSink {}

/// A plain synchronous grabber, for the clean reference wire.
struct FillGrabber {
    seq: u32,
}
impl FrameGrabber for FillGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        let len = buf.len();
        fill_frame(buf, self.seq);
        self.seq = self.seq.wrapping_add(1);
        Ok(len)
    }
}

/// A peripheral that latches a fault at [`FAULT_AT`] which only a `reset`
/// re-arms: the mid-stream sticky fault the supervisor must reset through.
struct StickyGrabber {
    seq: u32,
    armed: bool,
    tripped: bool,
}
impl FrameGrabber for StickyGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        if !self.armed {
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        if !self.tripped && self.seq == FAULT_AT {
            self.tripped = true;
            self.armed = false;
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        let len = buf.len();
        fill_frame(buf, self.seq);
        self.seq = self.seq.wrapping_add(1);
        Ok(len)
    }
    async fn reset(&mut self) -> Result<(), G2gError> {
        self.armed = true;
        Ok(())
    }
}

/// A peripheral that never recovers: every capture faults.
struct DeadGrabber;
impl FrameGrabber for DeadGrabber {
    async fn capture(&mut self, _buf: &mut [u8]) -> Result<usize, G2gError> {
        Err(G2gError::Hardware(HardwareError::Peripheral))
    }
}

/// A watchdog timer counting refreshes (the mock IWDG for the proof).
#[derive(Default)]
struct CountIwdg {
    feeds: u32,
}
impl WatchdogTimer for CountIwdg {
    fn feed(&mut self) {
        self.feeds = self.feeds.wrapping_add(1);
    }
}

/// The clean reference wire: the same pipeline with a non-faulting grabber.
fn reference() -> (u64, u32) {
    let cap_ring: StaticLendRing<1, CAP_BYTES> = StaticLendRing::new();
    let enc_ring: StaticLendRing<1, ULAW_BYTES> = StaticLendRing::new();
    // SAFETY: both rings outlive the run (the runner drains before they drop).
    let src = unsafe { GrabberSrc::with_ring(FillGrabber { seq: 0 }, &cap_ring, FRAME_NS) }
        .with_frame_limit(TARGET);
    // SAFETY: as above.
    let enc = unsafe { G711Enc::with_ring(Law::Mulaw, &enc_ring) };
    let mut sink = SumSink { sum: 0, count: 0 };
    let _ = drive_ready(run_source_transform_sink(src, enc, &mut sink));
    (sink.sum, sink.count)
}

#[entry]
fn main() -> ! {
    let (want_sum, want_count) = reference();

    // --- Scenario 1: recover through a latched mid-stream fault. ---
    let cap_ring: StaticLendRing<2, CAP_BYTES> = StaticLendRing::new();
    let enc_ring: StaticLendRing<2, ULAW_BYTES> = StaticLendRing::new();
    // SAFETY: both rings outlive this run.
    let src = unsafe {
        GrabberSrc::with_ring(
            StickyGrabber { seq: 0, armed: true, tripped: false },
            &cap_ring,
            FRAME_NS,
        )
    }
    .with_frame_limit(TARGET);
    // SAFETY: as above.
    let enc = unsafe { G711Enc::with_ring(Law::Mulaw, &enc_ring) };
    let mut sink = SumSink { sum: 0, count: 0 };
    let mut wd = SupervisorWatchdog::new(CountIwdg::default());
    let (report, outcome) =
        run_supervised(src, SinkChain(enc, &mut sink), RetryThenReset::default(), &mut wd);

    let recovered_ok = outcome == RunOutcome::Completed
        && sink.count == want_count
        && sink.sum == want_sum
        && report.resets == 1
        && !report.escalated
        && wd.feeds() == TARGET;

    // --- Scenario 2: escalate on a dead peripheral, within bounds. ---
    let cap_ring2: StaticLendRing<2, CAP_BYTES> = StaticLendRing::new();
    let enc_ring2: StaticLendRing<2, ULAW_BYTES> = StaticLendRing::new();
    // SAFETY: both rings outlive this run.
    let src2 = unsafe { GrabberSrc::with_ring(DeadGrabber, &cap_ring2, FRAME_NS) }
        .with_frame_limit(TARGET);
    // SAFETY: as above.
    let enc2 = unsafe { G711Enc::with_ring(Law::Mulaw, &enc_ring2) };
    let mut sink2 = SumSink { sum: 0, count: 0 };
    let mut wd2 = SupervisorWatchdog::new(CountIwdg::default());
    let (report2, outcome2) =
        run_supervised(src2, SinkChain(enc2, &mut sink2), RetryThenReset::new(2, 1), &mut wd2);

    let escalated_ok = matches!(outcome2, RunOutcome::Escalated(_))
        && report2.escalated
        && report2.faults == 4
        && sink2.count == 0
        && wd2.feeds() == 0;

    let ok = recovered_ok && escalated_ok;

    if let Ok(mut out) = hio::hstdout() {
        let mut line = [0u8; 96];
        let mut pos = 0;
        put_str(&mut line, &mut pos, "g2g-supervise: delivered=");
        put_u32(&mut line, &mut pos, sink.count);
        put_str(&mut line, &mut pos, " resets=");
        put_u32(&mut line, &mut pos, report.resets);
        put_str(&mut line, &mut pos, " wd=");
        put_u32(&mut line, &mut pos, wd.feeds());
        put_str(&mut line, &mut pos, " escalated=");
        put_u32(&mut line, &mut pos, report2.faults);
        put_str(&mut line, &mut pos, if ok { " OK\n" } else { " FAIL\n" });
        let _ = out.write_all(line.get(..pos).unwrap_or(&[]));
    }

    debug::exit(if ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    loop {}
}

/// Append `v` in decimal to `buf` at `pos` (no `core::fmt`).
fn put_u32(buf: &mut [u8], pos: &mut usize, v: u32) {
    let mut digits = [0u8; 10];
    let mut n = 0;
    let mut v = v;
    loop {
        if let Some(d) = digits.get_mut(n) {
            *d = b'0' + (v % 10) as u8;
        }
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        if let (Some(dst), Some(&src)) = (buf.get_mut(*pos), digits.get(n)) {
            *dst = src;
            *pos += 1;
        }
    }
}

fn put_str(buf: &mut [u8], pos: &mut usize, s: &str) {
    for &b in s.as_bytes() {
        if let Some(dst) = buf.get_mut(*pos) {
            *dst = b;
            *pos += 1;
        }
    }
}
