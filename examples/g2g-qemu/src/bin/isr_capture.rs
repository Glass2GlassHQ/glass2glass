//! ISR-driven capture proof (M651): the interrupt/DMA concurrency model on the
//! emulated Cortex-M4. A SysTick interrupt handler is the *producer*, running in
//! real interrupt context: each tick it fills a frame into a `static`
//! `SpscFrameRing` (standing in for a DMA-completion ISR marking a buffer ready).
//! The pipeline is the *consumer*, running in the main context: it drains the
//! ring through a real element chain (`SpscCaptureSrc -> G.711 mu-law encode ->
//! checksum`), sleeping on `wfi` between frames and waking on the capture
//! interrupt. Producer and consumer thus run in genuinely different execution
//! contexts, concurrently, handing frames across the ISR boundary lock-free.
//!
//! Correctness is proved by equivalence to synchronous delivery: the same frames
//! encoded the same way must yield the same wire. `reference()` runs the
//! identical pipeline with a plain synchronous grabber (no ISR); `main` runs it
//! fed by the SysTick ISR. Equal checksums (and the full `TARGET` frame count)
//! mean every interrupt-produced frame reached the pipeline in capture order,
//! uncorrupted. `tools/qemu-check.sh` boots this and asserts the exit code +
//! banner, upgrading "the SPSC ring is sound on the host" to "an ISR feeds a g2g
//! pipeline correctly on the Cortex-M ISA".

#![no_std]
#![no_main]

use core::sync::atomic::{AtomicU32, Ordering};

use cortex_m::peripheral::syst::SystClkSource;
use cortex_m::peripheral::Peripherals;
use cortex_m_rt::{entry, exception};
use cortex_m_semihosting::{debug, hio};

use g2g_core::staticpool::StaticLendRing;
use g2g_core::{
    drive_ready, run_source_transform_sink, step_source_sink, Frame, MemoryDomain, SinkChain,
    SpscCaptureSrc, SpscFrameRing, StaticSink, Step,
};
use g2g_mcu::{FrameGrabber, G711Enc, GrabberSrc, Law};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// 16 samples of mono S16LE per frame: 32 capture bytes, 16 mu-law bytes.
const SAMPLES: usize = 16;
const CAP_BYTES: usize = SAMPLES * 2;
const ULAW_BYTES: usize = SAMPLES;
/// Ring depth: a double-buffer plus slack (usable capacity 3).
const SLOTS: usize = 4;
/// Frames to capture and process.
const TARGET: u32 = 64;
/// Nominal frame period for the derived PTS (not the SysTick rate).
const FRAME_NS: u64 = 1_000_000;
/// SysTick reload: a comfortable margin over the per-frame processing cost, so
/// the consumer drains each frame before the next tick (no overrun by timing).
const SYST_RELOAD: u32 = 100_000;

/// The capture ring, shared between the SysTick ISR (producer) and main
/// (consumer). `const`-constructed, so it lives in a `static`, the DMA-ring idiom.
static RING: SpscFrameRing<SLOTS, CAP_BYTES> = SpscFrameRing::new();
/// The producer's next capture sequence number (written only by the ISR).
static PROD_SEQ: AtomicU32 = AtomicU32::new(0);

/// Fill one frame deterministically from its capture sequence: a signed ramp that
/// sweeps several G.711 segments. Shared by the ISR producer and the synchronous
/// reference grabber, so the two differ only in *how* the frame is delivered.
/// Panic-free (slice pattern, constant array indices) for the heap-free archive.
fn fill_frame(buf: &mut [u8; CAP_BYTES], seq: u32) {
    for (i, pair) in buf.chunks_exact_mut(2).enumerate() {
        let sample = (((seq.wrapping_add(i as u32)) & 0xff) as i32 - 128) * 128;
        let bytes = (sample as i16).to_le_bytes();
        if let [lo, hi] = pair {
            *lo = bytes[0];
            *hi = bytes[1];
        }
    }
}

/// The SysTick interrupt handler: the *producer*, in real interrupt context.
/// Each tick it publishes the next frame into the ring. It does not advance the
/// sequence when the ring is full (it retries the same frame next tick), so no
/// frame is lost to timing jitter, the consumer sees exactly `TARGET` frames in
/// order. (The drop-on-full back-pressure path is exercised by the host tests,
/// `g2g-mcu/tests/m651_isr_capture.rs`.)
#[exception]
fn SysTick() {
    let seq = PROD_SEQ.load(Ordering::Relaxed);
    if seq >= TARGET {
        return; // capture complete
    }
    if RING.produce(|buf| fill_frame(buf, seq)).is_ok() {
        PROD_SEQ.store(seq + 1, Ordering::Relaxed);
    }
}

/// A checksumming sink: sums every encoded byte and counts frames.
struct SumSink {
    sum: u64,
    count: u32,
}

impl StaticSink for SumSink {
    async fn consume(&mut self, frame: Frame) -> Result<(), g2g_core::error::G2gError> {
        if let MemoryDomain::System(s) = &frame.domain {
            for &b in s.as_slice() {
                self.sum = self.sum.wrapping_add(b as u64);
            }
        }
        self.count = self.count.wrapping_add(1);
        Ok(())
    }
}

/// A plain synchronous grabber over [`fill_frame`], for the reference run.
struct FillGrabber {
    seq: u32,
}

impl FrameGrabber for FillGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, g2g_core::error::G2gError> {
        let len = buf.len();
        if let Ok(arr) = <&mut [u8; CAP_BYTES]>::try_from(&mut *buf) {
            fill_frame(arr, self.seq);
        }
        self.seq = self.seq.wrapping_add(1);
        Ok(len)
    }
}

/// Run the SAME pipeline (`capture -> G.711 -> checksum`) with synchronous frame
/// delivery, for the reference wire the ISR-fed run must reproduce exactly.
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

    let Some(mut p) = Peripherals::take() else {
        debug::exit(debug::EXIT_FAILURE);
        loop {}
    };

    // The consumer pipeline: drain the ISR-filled ring through the encoder into
    // the checksum sink. The capture source sleeps on `wfi` while the ring is
    // empty, waking on the SysTick capture interrupt.
    let enc_ring: StaticLendRing<1, ULAW_BYTES> = StaticLendRing::new();
    let mut src =
        SpscCaptureSrc::new(&RING, || cortex_m::asm::wfi(), FRAME_NS).with_frame_limit(TARGET);
    // SAFETY: `enc_ring` outlives the pipeline below.
    let enc = unsafe { G711Enc::with_ring(Law::Mulaw, &enc_ring) };
    let mut sink = SumSink { sum: 0, count: 0 };
    let mut tail = SinkChain(enc, &mut sink);

    // Arm SysTick as a periodic interrupt: this starts the producer. Do it last,
    // right before the consume loop, so no ticks pile up during setup.
    let syst = &mut p.SYST;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(SYST_RELOAD);
    syst.clear_current();
    syst.enable_counter();
    syst.enable_interrupt();

    let mut consumed = 0u32;
    let ok = loop {
        match step_source_sink(&mut src, &mut tail) {
            Ok(Step::Advanced) => consumed = consumed.wrapping_add(1),
            Ok(Step::Eos) => break true,
            _ => break false,
        }
    };

    let sum_ok = ok && sink.sum == want_sum && sink.count == want_count && consumed == TARGET;

    if let Ok(mut out) = hio::hstdout() {
        let mut line = [0u8; 96];
        let mut pos = 0;
        put_str(&mut line, &mut pos, "g2g-isr: captured=");
        put_u32(&mut line, &mut pos, consumed);
        put_str(&mut line, &mut pos, " overruns=");
        put_u32(&mut line, &mut pos, RING.overruns());
        put_str(&mut line, &mut pos, if sum_ok { " OK\n" } else { " FAIL\n" });
        let _ = out.write_all(line.get(..pos).unwrap_or(&[]));
    }

    debug::exit(if sum_ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
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
