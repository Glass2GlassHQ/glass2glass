// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::clock::{CaptureAnchor, ClockSync};
use g2g_core::{G2gError, PropKind, PropertySpec};

pub(crate) const IS_LIVE_PROPERTY: PropertySpec = PropertySpec::new(
    "is-live",
    PropKind::Bool,
    "act as a live source: report live latency and produce on the pipeline clock",
)
.with_default("false");

pub(crate) async fn sleep_until(sync: &ClockSync, due_ns: u64) -> Result<(), G2gError> {
    let now_ns = sync.now_ns();
    if due_ns <= now_ns {
        return Ok(());
    }
    // Sleep on the clock's own timer when it has one, so the pacing follows
    // that timeline (and a test clock can skip the wait). An elected clock
    // that cannot sleep (an audio or PTP clock) still yields a deadline, so
    // wait out the interval on the tokio timer instead.
    if let Some(ticker) = sync.clock.as_ticker() {
        ticker.sleep_until_ns(due_ns).await;
        return Ok(());
    }
    #[cfg(feature = "std")]
    {
        tokio::time::sleep(core::time::Duration::from_nanos(due_ns - now_ns)).await;
        Ok(())
    }
    #[cfg(not(feature = "std"))]
    Err(G2gError::NotConfigured)
}

#[derive(Debug)]
pub(crate) struct LivePacer {
    sync: ClockSync,
    anchor: CaptureAnchor,
}

impl LivePacer {
    pub(crate) fn new(sync: ClockSync) -> Self {
        Self {
            sync,
            anchor: CaptureAnchor::new(),
        }
    }

    pub(crate) async fn wait_for(&mut self, stream_time_ns: u64) -> Result<u64, G2gError> {
        let running_time_ns = self.anchor.stamp(&self.sync, stream_time_ns, 0);
        sleep_until(&self.sync, self.sync.base_time() + running_time_ns).await?;
        Ok(running_time_ns)
    }
}
