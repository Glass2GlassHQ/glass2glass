//! Wall-clock implementation of [`PipelineClock`] / [`AsyncClock`] for std
//! targets. Backed by `std::time::Instant` and `tokio::time::sleep`.

use core::future::Future;
use core::pin::Pin;
use std::time::{Duration, Instant};

use alloc::boxed::Box;

use g2g_core::{AsyncClock, PipelineClock};

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
