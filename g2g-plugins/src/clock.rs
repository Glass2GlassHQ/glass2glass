//! Wall-clock implementation of [`PipelineClock`] / [`AsyncClock`] for std
//! targets. Backed by `std::time::Instant` and `tokio::time::sleep`.

use core::future::Future;
use core::pin::Pin;
use std::time::{Duration, Instant};

use alloc::boxed::Box;

use g2g_core::{AsyncClock, DynAsyncClock, PipelineClock};

#[derive(Debug, Clone, Copy)]
pub struct WallClock {
    epoch: Instant,
}

impl WallClock {
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }
}

impl Default for WallClock {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineClock for WallClock {
    fn now_ns(&self) -> u64 {
        // Saturate: u64 ns covers ~584 years; a pipeline runtime measured
        // from process start will not overflow this in practice.
        self.epoch
            .elapsed()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        Some(self)
    }
}

impl AsyncClock for WallClock {
    type SleepFuture<'a> = Pin<Box<dyn Future<Output = ()> + 'a>>;

    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> Self::SleepFuture<'a> {
        Box::pin(async move {
            let now = self.now_ns();
            if deadline_ns > now {
                tokio::time::sleep(Duration::from_nanos(deadline_ns - now)).await;
            }
        })
    }
}

/// Time of day as a [`PipelineClock`]: `now_ns` is nanoseconds since the UNIX
/// epoch, not a pipeline-relative timeline. Only for elements that display or
/// stamp civil time (`clockoverlay`); never hand it to the runner as the
/// pipeline clock, which needs a monotonic source ([`WallClock`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct UnixEpochClock;

impl PipelineClock for UnixEpochClock {
    fn now_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos().try_into().unwrap_or(u64::MAX))
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cooperative runner derives a fan-in element's deadline tick from the
    /// pipeline clock (M880), so the clock every `g2g-launch` pipeline runs against
    /// has to offer itself as that timer, on the same timeline it reports.
    #[test]
    fn the_wall_clock_offers_itself_as_a_ticker() {
        let clock = WallClock::new();
        let ticker = PipelineClock::as_ticker(&clock).expect("WallClock can sleep on a deadline");
        assert!(ticker.now_ns() <= clock.now_ns(), "same timeline");
    }
}
