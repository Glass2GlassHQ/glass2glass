//! Per-frame timing / jitter report for the flagship audio graph (M645):
//! run `capture -> convert -> resample -> mix -> encode -> RTP` on the
//! emulated Cortex-M4 with a SysTick stamp on every emitted packet, and
//! report the per-frame execution cost over semihosting. `tools/
//! timing-report.sh` boots this under QEMU `-icount shift=0,sleep=off`,
//! where virtual time is a pure function of the executed instruction
//! stream: the numbers are deterministic (the script runs it twice and
//! asserts identical output) and budget-enforced, the timing sibling of the
//! footprint report.
//!
//! The first frame carries the one-time cost (caps negotiation, resampler
//! history warm-up) and is reported separately; frames 2..N are the steady
//! state whose spread is the jitter. SysTick runs at the emulated 25 MHz
//! system clock (MPS2-AN386), so ticks convert to microseconds at /25.
//!
//! No `core::fmt`: numbers are printed with a small manual decimal writer,
//! keeping this binary as thin over the proven pipeline as the checksum one.

#![no_std]
#![no_main]

use cortex_m::peripheral::syst::SystClkSource;
use cortex_m::peripheral::{Peripherals, SYST};
use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};
use noalloc_pipeline::audio::{run_audio_with, TimedSumSender, AUDIO_EXPECTED_CHECKSUM};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// SysTick reload: full 24-bit range, so per-frame deltas (well under 2^24
/// cycles) never alias across a wrap.
const SYST_RELOAD: u32 = 0x00FF_FFFF;

/// Elapsed ticks between two reads of the down-counting SysTick.
fn elapsed(from: u32, to: u32) -> u32 {
    from.wrapping_sub(to) & SYST_RELOAD
}

/// Append `v` in decimal to `buf` at `pos` (no core::fmt).
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

#[entry]
fn main() -> ! {
    let Some(p) = Peripherals::take() else {
        // Unreachable on a fresh boot; fail loud through the exit code.
        debug::exit(debug::EXIT_FAILURE);
        loop {}
    };
    let mut syst = p.SYST;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(SYST_RELOAD);
    syst.clear_current();
    syst.enable_counter();

    let start = SYST::get_current();
    let sender = noalloc_pipeline::drive_ready(run_audio_with(TimedSumSender::new(
        SYST::get_current,
    )));
    let Some(sender) = sender else {
        debug::exit(debug::EXIT_FAILURE);
        loop {}
    };
    let ok = sender.checksum() == AUDIO_EXPECTED_CHECKSUM;
    let stamps = sender.stamps();

    // First frame = one-time cost (negotiation + warm-up); the rest are the
    // steady state: min / max (the WCET) and jitter (max - min).
    let mut first = 0u32;
    let mut min = u32::MAX;
    let mut max = 0u32;
    let mut prev = start;
    for (i, &stamp) in stamps.iter().enumerate() {
        let d = elapsed(prev, stamp);
        prev = stamp;
        if i == 0 {
            first = d;
        } else {
            if d < min {
                min = d;
            }
            if d > max {
                max = d;
            }
        }
    }
    if min == u32::MAX {
        min = 0;
    }

    let mut line = [0u8; 128];
    let mut pos = 0;
    put_str(&mut line, &mut pos, "g2g-timing: frames=");
    put_u32(&mut line, &mut pos, stamps.len() as u32);
    put_str(&mut line, &mut pos, " first=");
    put_u32(&mut line, &mut pos, first);
    put_str(&mut line, &mut pos, " steady_min=");
    put_u32(&mut line, &mut pos, min);
    put_str(&mut line, &mut pos, " steady_max=");
    put_u32(&mut line, &mut pos, max);
    put_str(&mut line, &mut pos, " jitter=");
    put_u32(&mut line, &mut pos, max.saturating_sub(min));
    put_str(&mut line, &mut pos, " ticks\n");
    if let Ok(mut out) = hio::hstdout() {
        let _ = out.write_all(line.get(..pos).unwrap_or(&[]));
        if !ok {
            let _ = out.write_all(b"g2g-timing: FAIL, wrong checksum\n");
        }
    }
    debug::exit(if ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    loop {}
}
