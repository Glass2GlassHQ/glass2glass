//! Embassy RTOS clock backend: [`PipelineClock`] / [`AsyncClock`] over
//! `embassy-time`, the no_std analog of `WallClock` (std/tokio) and `WasmClock`
//! (browser). `now_ns` reads `embassy_time::Instant`; `sleep_until_ns` returns
//! an `embassy_time::Timer` directly (no allocation), suiting strict no-heap
//! targets.
//!
//! `embassy-time` needs a HAL-provided time driver registered at link; the
//! clock is verified by a bare-metal compile, with the driver owed to hardware.

use embassy_time::{Duration, Instant, Timer};

use g2g_core::{AsyncClock, PipelineClock};

#[derive(Debug, Clone, Copy)]
pub struct EmbassyClock {
    /// `Instant::now()` captured at construction, so `now_ns` starts near zero
    /// like `WallClock`'s epoch.
    epoch: Instant,
}

impl EmbassyClock {
    pub fn new() -> Self {
        Self { epoch: Instant::now() }
    }
}

impl Default for EmbassyClock {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineClock for EmbassyClock {
    fn now_ns(&self) -> u64 {
        // embassy-time resolves to microseconds; sub-us precision is bounded by
        // the configured tick rate.
        (Instant::now() - self.epoch).as_micros().saturating_mul(1000)
    }
}

impl AsyncClock for EmbassyClock {
    type SleepFuture<'a> = Timer;

    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> Self::SleepFuture<'a> {
        // Timer::at resolves immediately when the deadline is already past.
        Timer::at(self.epoch + Duration::from_micros(deadline_ns / 1000))
    }
}
