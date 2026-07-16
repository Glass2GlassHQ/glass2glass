//! Hardware watchdog seam for the supervised MCU path: the thin trait a board's
//! watchdog peripheral implements, plus the adapter that lets the
//! [`Supervisor`](g2g_core::supervise) pet it on every frame of forward progress.
//!
//! `embedded-hal` 1.0 dropped the watchdog trait it carried in 0.2, so this is a
//! local seam (like [`FrameGrabber`](crate::FrameGrabber) and
//! [`PacketSender`](crate::PacketSender)): a one-method trait a vendor HAL's
//! independent-watchdog refresh (`IWDG.refresh()`, `wdt.feed()`) satisfies with a
//! trivial adapter, so the supervisor's liveness logic is host-testable with a
//! mock and a board port is only the register poke.
//!
//! The supervisor pets on each delivered frame and *stops* petting when the
//! pipeline wedges or escalates; the hardware watchdog then times out and resets
//! the MCU. That is the last-resort recovery the safety market requires: even a
//! bug that hangs the pipeline outside the fault-return path (a stuck interrupt,
//! a spin in a driver) is caught, because no frame means no refresh.

use g2g_core::supervise::Watchdog;

/// A board's watchdog timer: `feed` refreshes it before it expires. A vendor
/// HAL's independent-watchdog reload satisfies this directly.
pub trait WatchdogTimer {
    /// Refresh the watchdog countdown (the STM32 `IWDG` key reload, an RTOS
    /// software-watchdog kick).
    fn feed(&mut self);
}

/// Adapts a [`WatchdogTimer`] into the supervisor's [`Watchdog`], so a hardware
/// watchdog is petted once per delivered frame: forward progress refreshes it,
/// and a stalled or escalated pipeline (no delivered frame) lets it expire and
/// reset the chip.
#[derive(Debug, Default, Clone, Copy)]
pub struct SupervisorWatchdog<W> {
    timer: W,
    /// Total refreshes issued, for the fault-accounting / test assertion (a real
    /// deployment ignores it).
    feeds: u32,
}

impl<W: WatchdogTimer> SupervisorWatchdog<W> {
    /// Wrap a board watchdog timer for the supervisor.
    pub const fn new(timer: W) -> Self {
        Self { timer, feeds: 0 }
    }

    /// How many times the watchdog has been fed (once per delivered frame).
    pub fn feeds(&self) -> u32 {
        self.feeds
    }

    /// Release the underlying timer.
    pub fn free(self) -> W {
        self.timer
    }
}

impl<W: WatchdogTimer> Watchdog for SupervisorWatchdog<W> {
    fn pet(&mut self) {
        self.feeds = self.feeds.saturating_add(1);
        self.timer.feed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock IWDG counting refreshes.
    #[derive(Default)]
    struct MockIwdg {
        refreshes: u32,
    }
    impl WatchdogTimer for MockIwdg {
        fn feed(&mut self) {
            self.refreshes += 1;
        }
    }

    #[test]
    fn petting_the_supervisor_watchdog_refreshes_the_timer() {
        let mut wd = SupervisorWatchdog::new(MockIwdg::default());
        wd.pet();
        wd.pet();
        wd.pet();
        assert_eq!(wd.feeds(), 3, "supervisor counted three progress pets");
        assert_eq!(wd.free().refreshes, 3, "each pet reached the hardware timer");
    }
}
